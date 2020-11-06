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
import keyboard

with open("firebase_config.json", 'r') as f:
    firebase_config = json.load(f)

pars = Parsing()

pre = Preprocessing()

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
user = auth.sign_in_with_email_and_password(firebase_config["credential"]["email"], firebase_config["credential"]["password"])

terminate = False

async def shutdown(index, url, duration, cam_timeout):
    status = False
    headers = {'time': str(duration)}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cam_timeout)) as session:
        try:
            async with session.get(url[index]+"/shutdown", headers=headers) as response:
                if (response.status == 200):
                    print(f"Response status ({url[index]}): {response.status}")
                    status = True
    
            
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {url[index]} {http_err}")
        
        except aiohttp.ClientConnectorError as e:
            print(f'Connection Error {url[index]} {str(e)}')
            
        except Exception as err:
            print(f"An error ocurred: {url[index]} {err}")
        
        return index, status, 0


async def capture(index, url, mask, cam_timeout, free_threshold):
    status = False
    free_space = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=cam_timeout)) as session:
        try:
            async with session.get(url[index]+"/capture") as response:
                # response.raise_for_status()
                if (response.status == 200):
                    print(f"Response status ({url[index]}): {response.status}")
                    image = np.asarray(bytearray(await response.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    status = True
    
            
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {url[index]} {http_err}")
        
        except aiohttp.ClientConnectorError as e:
            print(f'Connection Error {url[index]} {str(e)}')
            
        except Exception as err:
            print(f"An error ocurred: {url[index]} {err}")
        
        # image = np.zeros((1600, 1200, 3), dtype="uint8")
        # time.sleep(random.uniform(0.5, 1.0))

        if status:
            pre.setMask(mask[index])
            pre.setImage(image)
            crop = pre.getCrop()

            for frame in crop:
                frame = cv2.resize(frame, (input_shape[1], input_shape[2]), interpolation = cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32)
                frame = frame / 255.
                frame = np.expand_dims(frame, 0)

                interpreter.set_tensor(input_details[0]['index'], frame)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                if (output[0][1] > free_threshold):
                    free_space += 1

        return index, status, free_space
        
async def capture_request(indexs, url, mask, cam_timeout, free_threshold):
    result = await asyncio.gather(*[capture(i, url, mask, cam_timeout, free_threshold) for i in indexs])
    return result

async def shutdown_request(indexs, url, duration, cam_timeout):
    result = await asyncio.gather(*[shutdown(i, url, duration, cam_timeout) for i in indexs])
    return result

def start_request(req_type, indexs, url, mask=None, duration=None, cam_timeout=None, free_threshold=None):
    loop = asyncio.get_event_loop()
    if (req_type == "capture"):
        return loop.run_until_complete(capture_request(indexs, url, mask, cam_timeout, free_threshold))
    elif (req_type == "shutdown"):
        return loop.run_until_complete(shutdown_request(indexs, url, duration, cam_timeout))

def get_hour_to(end):
    now = time.localtime().tm_hour
    delta = end - now
    if delta < 0:
        return 24 - abs(delta)
    else:
        return delta

def main():
    cam_addr_list = pars.get_url()
    maskParam = pars.get_masking()

    NUM_CORES = cpu_count() # Our number of CPU cores (including logical cores)
    NUM_URL = len(cam_addr_list)
    URL_PER_CORE = floor(NUM_URL / NUM_CORES)
    REMAINDER = NUM_URL % NUM_CORES

    # print("url count: {0} cpu count: {1} count {2} remainder {3}".format(NUM_URL, NUM_CORES, URL_PER_CORE, REMAINDER))

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
                # print("j{0} : {1}".format(i, j))
                indexs.append(j)

            # print("core {0} get {1} task from {2} to {3}".format(i, len(indexs), start, stop))
            
            new_future = executor.submit(
                start_request, # Function to perform
                # v Arguments v
                "capture",
                indexs,
                cam_addr_list,
                mask=maskParam,
                cam_timeout=pars.get_cam_timeout(),
                free_threshold=pars.get_free_threshold()
            )
            futures.append(new_future)

    concurrent.futures.wait(futures)

    for future in futures:
        # print(future.result())
        for f in future.result():
            result.append(f)
    
    print(f"result: {result}")
    pars.input_status(result)
    # print(pars.get_free_lot_all())

def exit_callback(e):
    global terminate
    print("Terminate program")
    terminate = True

if __name__ == "__main__":

    camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
    systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
    
    while not pars.config_ready():
        print("wait for config...\n")
        time.sleep(1)
    
    print(f"start_time: {pars.get_start_time()}\nend_time: {pars.get_end_time()}\nupdate_rate: {pars.get_update_rate()}\ncam_timeout: {pars.get_cam_timeout()}\nfree_threshold: {pars.get_free_threshold()}")
    all_sleep = False
    last = time.time()
    keyboard.on_press_key("q", exit_callback)

    while not terminate:
        if time.time() > (last + 3000):
            last = time.time()
            user = auth.refresh(user['refreshToken'])
        
        if not (time.localtime().tm_hour >= pars.get_end_time() or time.localtime().tm_hour <= pars.get_start_time()):
            print("work time")
            all_sleep = False
            start = time.time()
            print("start process")
            print(f"Process time is {timeit.timeit(main, number=1)}")
            db.child("free_space").set(pars.get_free_lot_all(), user['idToken'])
            end = time.time()
            print(f"Total time for update: {end-start}")
            if ((end-start) < pars.get_update_rate()):
                time.sleep(pars.get_update_rate() - (end-start))
            
        elif not all_sleep:
            print("Shutdown start")
            duration = get_hour_to(pars.get_start_time()-1)
            if not pars.all_cam_true():
                duration = get_hour_to(pars.get_start_time()-1)
                print(f"Sleep duration {duration} Hour")
                res = start_request("shutdown", range(len(pars.get_url())), pars.get_url(), duration=duration, cam_timeout=pars.get_cam_timeout())
                pars.input_status(res)
                db.child("free_space").set(pars.get_free_lot_all(), user['idToken'])
                time.sleep(5)
            else:
                all_sleep = True
                print("system sleep")
    
    print("Program shutdown")
    keyboard.unhook_all()
    camConfig_stream.close()
    systemConfig_stream.close()
