import aiohttp                          # For asynchronously making HTTP requests
import asyncio
import concurrent.futures               # Allows creating new processes
from multiprocessing import cpu_count   # Returns our number of CPU cores
                                        # Helps divide up our requests evenly across our CPU cores
from math import floor
import numpy as np
import cv2
import requests
import timeit
import tflite_runtime.interpreter as tflite
from Tools import Preprocessing
from Tools import Parsing
import json
import sys
import time
import random
import pyrebase
from pynput import keyboard
import dpath.util
import logging

logging.getLogger().setLevel(logging.INFO)

with open("firebase_config.json", 'r') as f:
    firebase_config = json.load(f)

pars = Parsing()

model_path = "./model.tflite"

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

config = firebase_config["config"]
firebase = pyrebase.initialize_app(config)

db = firebase.database()
auth = firebase.auth()

terminate = False

def clahe(img):
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final

async def shutdown(session, index, url, slot_path, duration, cam_timeout):
    status = False
    result = {}
    headers = {'time': str(duration)}
    try:
        async with session.get(url[index]+"/shutdown", headers=headers) as response:
            if (response.status == 200):
                logging.info(f"Response status ({url[index]}): {response.status}")
                status = True

        
    except requests.exceptions.HTTPError as http_err:
        logging.info(f"HTTP error occurred: {url[index]} {http_err}")
    
    except aiohttp.ClientConnectorError as e:
        logging.info(f'Connection Error {url[index]} {str(e)}')
        
    except Exception as err:
        logging.info(f"An error ocurred: {url[index]} {err}")
    
    path = slot_path[index][0].split('/')[:-2]
    path.append('status')
    dpath.util.new(result, path, status)
    
    return index, result, 0

async def capture(session, index, url, slot_path, slot_reserved, mask, cam_timeout, free_threshold):
    pre = Preprocessing()
    status = False
    result = {}
    total_free = 0
    
    # try:
    #     async with session.get(url[index]+"/capture") as response:
    #         # response.raise_for_status()
    #         if (response.status == 200):
    #             logging.info(f"Response status ({url[index]}): {response.status}")
    #             image = np.asarray(bytearray(await response.read()), dtype="uint8")
    #             image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    #             status = True

        
    # except requests.exceptions.HTTPError as http_err:
    #     logging.info(f"HTTP error occurred: {url[index]} {http_err}")
    
    # except aiohttp.ClientConnectorError as e:
    #     logging.info(f'Connection Error {url[index]} {str(e)}')
        
    # except Exception as err:
    #     logging.info(f"An error ocurred: {url[index]} {err}")
        
    image = np.zeros((1200, 1600, 3), dtype="uint8")
    # time.sleep(random.uniform(0.7, 1.0))
    time.sleep(1.0)
    status = True

    if status:
        pre.setMask(mask[index])
        pre.setImage(clahe(image))
        crop = pre.getCrop()

        for i,frame in enumerate(crop):
            frame = cv2.resize(frame, (input_shape[1], input_shape[2]), interpolation = cv2.INTER_CUBIC)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
            frame = frame / 255.
            frame = np.expand_dims(frame, 0)

            interpreter.set_tensor(input_details[0]['index'], frame)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            if (output[0][1] > free_threshold):
                # free_space.append(True)
                dpath.util.new(result, slot_path[index][i]+'/free', True)
                if not slot_reserved[index][i]:
                    total_free += 1
            else:
                # free_space.append(False)
                dpath.util.new(result, slot_path[index][i]+'/free', False)

        path = slot_path[index][0].split('/')[:-2]
        path.append('free')
        dpath.util.new(result, path, total_free)

    path = slot_path[index][0].split('/')[:-2]
    path.append('status')
    dpath.util.new(result, path, status)
    return index, result, total_free
        
async def capture_request(indexs, url, slot_path, slot_reserved, mask, cam_timeout, free_threshold):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cam_timeout)) as session:
        result = await asyncio.gather(*[capture(session, i, url, slot_path, slot_reserved, mask, cam_timeout, free_threshold) for i in indexs])
        await session.close()

    return result

async def shutdown_request(indexs, url, slot_path, duration, cam_timeout):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cam_timeout)) as session:    
        result = await asyncio.gather(*[shutdown(session, i, url, slot_path, duration, cam_timeout) for i in indexs])
        await session.close()

    return result

def start_request(req_type, indexs, url, slot_path=None, slot_reserved=None, mask=None, duration=None, cam_timeout=None, free_threshold=None):
    loop = asyncio.get_event_loop()
    if (req_type == "capture"):
        return loop.run_until_complete(capture_request(indexs, url, slot_path, slot_reserved, mask, cam_timeout, free_threshold))
    elif (req_type == "shutdown"):
        return loop.run_until_complete(shutdown_request(indexs, url, slot_path, duration, cam_timeout))

def get_hour_to(end):
    now = time.localtime().tm_hour
    delta = end - now
    if delta < 0:
        return 24 - abs(delta)
    else:
        return delta

def work_single_core():
    cam_addr_list = pars.get_url()
    maskParam = pars.get_masking()

    result = start_request("capture", range(len(cam_addr_list)), cam_addr_list, mask=maskParam, slot_path=pars.get_slot_path(), slot_reserved=pars.get_slot_reserved(), cam_timeout=pars.get_cam_timeout(), free_threshold=pars.get_free_threshold())

    logging.info(f"result: {result}")
    pars.input_status(result)

def work():
    logging.info("start work")
    cam_addr_list = pars.get_url()
    maskParam = pars.get_masking()
    timeout = pars.get_cam_timeout()
    free_threshold = pars.get_free_threshold()
    slot_path = pars.get_slot_path()
    slot_reserved = pars.get_slot_reserved()

    NUM_CORES = cpu_count()
    # NUM_CORES = 1
    NUM_URL = len(cam_addr_list)
    URL_PER_CORE = floor(NUM_URL / NUM_CORES)
    REMAINDER = NUM_URL % NUM_CORES

    # logging.info("url count: {0} cpu count: {1} count {2} remainder {3}".format(NUM_URL, NUM_CORES, URL_PER_CORE, REMAINDER))

    futures = [] # To store our futures
    result = []

    with concurrent.futures.ProcessPoolExecutor(NUM_CORES) as executor:
        for i in range(NUM_CORES):
            indexs = []
            if (i < REMAINDER):
                start = i * (URL_PER_CORE + 1)
                stop = start + (URL_PER_CORE + 1)
                
            else:
                start = (i * URL_PER_CORE) + REMAINDER
                stop = start + URL_PER_CORE


            for j in range(start, stop):
                # logging.info("j{0} : {1}".format(i, j))
                indexs.append(j)

            # logging.info("core {0} get {1} task from {2} to {3}".format(i, len(indexs), start, stop))
            
            new_future = executor.submit(
                start_request, # Function to perform
                # v Arguments v
                "capture",
                indexs,
                cam_addr_list,
                slot_path=slot_path,
                slot_reserved=slot_reserved,
                mask=maskParam,
                cam_timeout=timeout,
                free_threshold=free_threshold
            )
            futures.append(new_future)

    concurrent.futures.wait(futures)

    for future in futures:
        # logging.info(future.result())
        for f in future.result():
            result.append(f)
    logging.info("inference done, input result")
    # logging.info(f"result: {result}")
    # pars.input_status(result)
    try:
        logging.info(f"Input time is {timeit.timeit(lambda: pars.input_status(result), globals=globals(), number=1)}")
    except:
        logging.info("input status failed")
    
    logging.info("work done")


def on_press(key):
    global terminate
    try:
        if key.char == 'q':
            terminate = True
            logging.info("Terminate program")
    except AttributeError:
        pass
    finally:
        return

if __name__ == "__main__":
    
    connect = False
    while not connect:
        try:
            user = auth.sign_in_with_email_and_password(firebase_config["credential"]["email"], firebase_config["credential"]["password"])
            camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
            systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
            connect = True
        except:
            logging.info("connection to database failed")
            time.sleep(1)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not pars.config_ready() and not terminate:
        logging.info("wait for config...\n")
        time.sleep(2)
    
    # logging.info(f"start_time: {pars.get_start_time()}, end_time: {pars.get_end_time()}, update_rate: {pars.get_update_rate()}, cam_timeout: {pars.get_cam_timeout()}, free_threshold: {pars.get_free_threshold()}")
    all_sleep = False
    last = time.time()

    while not terminate:
        start = time.time()

        if time.time() > (last + 3000):
            last = time.time()
            try:
                camConfig_stream.close()
                systemConfig_stream.close()
                user = auth.refresh(user['refreshToken'])
            except:
                logging.info("refresh token failed")
            camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
            systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
        
        if not camConfig_stream.thread.is_alive():
            logging.info("camConfig_stream is dead")
            try:
                camConfig_stream.close()
            except Exception:
                # client.captureException(tags={'handled_status': 'catched_and_logged'})
                logging.info("close stream camConfig failed")
            camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
        
        if not systemConfig_stream.thread.is_alive():
            logging.info("systemConfig_stream is dead")
            try:
                systemConfig_stream.close()
            except Exception:
                # client.captureException(tags={'handled_status': 'catched_and_logged'})
                logging.info("close stream systemConfig failed")
            systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
        
        if not (time.localtime().tm_hour >= pars.get_end_time() or time.localtime().tm_hour < pars.get_start_time()):
            logging.info("work time")
            all_sleep = False
            logging.info(f"Process time is {timeit.timeit(work, number=1)}")
            # logging.info(f"Process time is {timeit.timeit(work_single_core, number=1)}")
            # db.child("free_space").set(pars.get_free(), user['idToken'])
            # logging.info(f"Upload time is {timeit.timeit("db.child("free_space").set(pars.get_free(), user['idToken'])", number=1)}")
            try:
                db.child("free_space").set(pars.get_free(), user['idToken'])
            except:
                logging.info("update upload failed")
            
        elif not all_sleep:
            logging.info("Sleep time")
            duration = get_hour_to(pars.get_start_time()-1)
            logging.info(f"Sleep duration {duration} Hour")
            res = start_request("shutdown", list(range(len(pars.get_url()))), pars.get_url(), slot_path=pars.get_slot_path(), duration=duration, cam_timeout=pars.get_cam_timeout())
            pars.input_status(res)
            try:
                db.child("free_space").set(pars.get_free(), user['idToken'])
            except:
                logging.info("update upload failed")
            logging.info("system sleep")
            all_sleep = True
        
        end = time.time()
        logging.info(f"Total time for update: {end-start}")
        
        if ((end-start) < pars.get_update_rate()):
            time.sleep(pars.get_update_rate() - (end-start))
    
    logging.info("Program shutdown")
    camConfig_stream.close()
    systemConfig_stream.close()
