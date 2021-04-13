import aiohttp                          
import asyncio
import concurrent.futures               
from multiprocessing import cpu_count   
                                        
from math import floor
import numpy as np
import cv2
import requests
import timeit
import tflite_runtime.interpreter as tflite
from Tools import Preprocessing
from Tools import Parsing
import json
import time
import pyrebase
from pynput import keyboard
import dpath.util
import logging
import argparse

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
        logging.error(f"HTTP error occurred: {url[index]} {http_err}")
    
    except aiohttp.ClientConnectorError as e:
        logging.error(f'Connection Error {url[index]} {str(e)}')
        
    except Exception as err:
        logging.error(f"An error ocurred: {url[index]} {err}")
    
    path = slot_path[index][0].split('/')[:-2]
    path.append('status')
    dpath.util.new(result, path, status)
    
    return index, result, 0

async def capture(session, index, url, slot_path, slot_reserved, mask, cam_timeout, threshold):
    pre = Preprocessing()
    status = False
    result = {}
    total_free = 0
    
    try:
        async with session.get(url[index]+"/capture") as response:
            if (response.status == 200):
                logging.info(f"Response status ({url[index]}): {response.status}")
                image = np.asarray(bytearray(await response.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                status = True

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {url[index]} {http_err}")
    
    except aiohttp.ClientConnectorError as e:
        logging.error(f'Connection Error {url[index]} {str(e)}')
        
    except Exception as err:
        logging.error(f"An error ocurred: {url[index]} {err}")

    if status:
        pre.setMask(mask[index])
        pre.setImage(image)
        crop = pre.getCrop()

        for i,frame in enumerate(crop):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (input_shape[1], input_shape[2]), interpolation = cv2.INTER_CUBIC)
            frame = frame.astype(np.float32)
            frame = frame / 255.
            frame = np.expand_dims(frame, 0)

            interpreter.set_tensor(input_details[0]['index'], frame)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            if (output[0][0] <= threshold):
                dpath.util.new(result, slot_path[index][i]+'/free', True)
                if not slot_reserved[index][i]:
                    total_free += 1
            else:
                dpath.util.new(result, slot_path[index][i]+'/free', False)

        path = slot_path[index][0].split('/')[:-2]
        path.append('free')
        dpath.util.new(result, path, total_free)

    path = slot_path[index][0].split('/')[:-2]
    path.append('status')
    dpath.util.new(result, path, status)
    return index, result, total_free
        
async def capture_request(indexs, url, slot_path, slot_reserved, mask, cam_timeout, threshold):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cam_timeout)) as session:
        result = await asyncio.gather(*[capture(session, i, url, slot_path, slot_reserved, mask, cam_timeout, threshold) for i in indexs])
        await session.close()

    return result

async def shutdown_request(indexs, url, slot_path, duration, cam_timeout):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cam_timeout)) as session:    
        result = await asyncio.gather(*[shutdown(session, i, url, slot_path, duration, cam_timeout) for i in indexs])
        await session.close()

    return result

def start_request(req_type, indexs, url, slot_path=None, slot_reserved=None, mask=None, duration=None, cam_timeout=None, threshold=None):
    loop = asyncio.get_event_loop()
    if (req_type == "capture"):
        return loop.run_until_complete(capture_request(indexs, url, slot_path, slot_reserved, mask, cam_timeout, threshold))
    elif (req_type == "shutdown"):
        return loop.run_until_complete(shutdown_request(indexs, url, slot_path, duration, cam_timeout))

def get_hour_to(end):
    now = time.localtime().tm_hour
    delta = end - now
    if delta < 0:
        return 24 - abs(delta)
    else:
        return delta

def sleep(user):
    duration = get_hour_to(pars.get_start_time()-1)
    logging.debug(f"Sleep duration {duration} Hour")
    res = start_request("shutdown", list(range(len(pars.get_url()))), pars.get_url(), slot_path=pars.get_slot_path(), duration=duration, cam_timeout=pars.get_cam_timeout())
    pars.input_status(res)
    try:
        db.child("free_space").set(pars.get_free(), user['idToken'])
    except:
        logging.error("upload to database failed")

def work(user):
    logging.info("initiate update process")
    cam_addr_list = pars.get_url()
    maskParam = pars.get_masking()
    timeout = pars.get_cam_timeout()
    threshold = pars.get_threshold()
    slot_path = pars.get_slot_path()
    slot_reserved = pars.get_slot_reserved()

    NUM_CORES = cpu_count()
    NUM_URL = len(cam_addr_list)
    URL_PER_CORE = floor(NUM_URL / NUM_CORES)
    REMAINDER = NUM_URL % NUM_CORES

    futures = []
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
                indexs.append(j)
            
            new_future = executor.submit(
                start_request,
                "capture",
                indexs,
                cam_addr_list,
                slot_path=slot_path,
                slot_reserved=slot_reserved,
                mask=maskParam,
                cam_timeout=timeout,
                threshold=threshold
            )
            futures.append(new_future)

    concurrent.futures.wait(futures)

    for future in futures:
        for f in future.result():
            result.append(f)
    
    logging.debug("inference done, generate result")
    logging.debug(f"result: {result}")
    start = time.time()
    pars.input_status(result)
    logging.debug(f"Input time is {time.time() - start} second")

    try:
        start = time.time()
        db.child('free_space').set(pars.get_free(), user['idToken'])
        logging.debug(f"upload time is {time.time() - start} second")
    except:
        logging.error("upload to database failed")
    
    logging.info("update finished")


def on_press(key):
    global terminate
    try:
        if key.char == 'x':
            terminate = True
            logging.info("Terminating program")
    except AttributeError:
        pass
    finally:
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-log", 
        "--log", 
        default="info",
        help=(
            "Provide logging level. "
            "Example --log debug', default='info'"
            )
    )

    options = parser.parse_args()
    levels = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warn': logging.WARNING,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG
    }
    level = levels.get(options.log.lower())
    if level is None:
        raise ValueError(
            f"log level given: {options.log}"
            f" -- must be one of: {' | '.join(levels.keys())}")
    logging.basicConfig(format='%(levelname)s:%(message)s', level=level)

    connect = False
    while not connect:
        try:
            user = auth.sign_in_with_email_and_password(firebase_config["credential"]["email"], firebase_config["credential"]["password"])
            camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
            systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
            connect = True
        except:
            logging.critical("connection to database failed")
            time.sleep(2)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not pars.config_ready() and not terminate:
        logging.info("wait for config...")
        time.sleep(2)
    
    logging.info("System config ready")
    logging.debug(f"System config is >>> start_time: {pars.get_start_time()}, end_time: {pars.get_end_time()}, threshold: {pars.get_threshold()}, update_rate: {pars.get_update_rate()}, cam_timeout: {pars.get_cam_timeout()}")

    sleep_mode = False
    last = time.time()

    while not terminate:
        update_time = 0.0
        logging.debug(f"Local time : {time.strftime('%b %d %Y %H:%M:%S', time.localtime())}")
        if not (time.localtime().tm_hour >= pars.get_end_time() or time.localtime().tm_hour < pars.get_start_time()):
            logging.info("work time")
            sleep_mode = False
            update_time = timeit.timeit(lambda: work(user), globals=globals(), number=1)
            logging.debug(f"Total time for update is {update_time}")
            
        elif not sleep_mode:
            logging.info("Sleep time")
            update_time = timeit.timeit(lambda: sleep(user), globals=globals(), number=1)
            sleep_mode = True
        
        if time.time() > (last + 3000):
            last = time.time()
            try:
                camConfig_stream.close()
                systemConfig_stream.close()
                user = auth.refresh(user['refreshToken'])
            except:
                logging.error("refresh token failed")

            connect = False
            while not connect:
                try:
                    camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
                    systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
                    connect = True
                except:
                    logging.critical("connection to database failed")
                    time.sleep(2)
        
        if not camConfig_stream.thread.is_alive():
            logging.warning("camConfig_stream is dead")
            try:
                camConfig_stream.close()
            except Exception:
                logging.error("close stream camConfig failed")

            camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
        
        if not systemConfig_stream.thread.is_alive():
            logging.warning("systemConfig_stream is dead")
            try:
                systemConfig_stream.close()
            except Exception:
                logging.error("close stream systemConfig failed")

            systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
        
        if (update_time < pars.get_update_rate()):
            time.sleep(pars.get_update_rate() - update_time)
    
    logging.info("Shutdown...")
    camConfig_stream.close()
    systemConfig_stream.close()
