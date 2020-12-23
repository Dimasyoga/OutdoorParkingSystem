import numpy as np
import time
import cv2
import tflite_runtime.interpreter as tflite
from Tools import Preprocessing
from Tools import Parsing
from urllib.request import urlopen
import requests
import urllib.error
from socket import timeout
import pyrebase
import json
from pynput import keyboard
import logging

logging.getLogger().setLevel(logging.INFO)

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

terminate = False

def capture(url, mask, reserved, cam_timeout, threshold):

    try:
        print(url)
        resp = urlopen(url+'/capture', timeout=cam_timeout)
        if resp.status == 200:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            print('get image')
        else:
            image = np.ones((1200, 1600, 3), dtype="uint8")    
            print('failed')

    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        image = np.ones((1200, 1600, 3), dtype="uint8")
        print(e.__dict__)
    except timeout:
        image = np.ones((1200, 1600, 3), dtype="uint8")
        print("timeout")

    pre.setMask(mask)
    pre.setImage(image)
    crop = pre.getCrop()
    pred = []

    for i,frame in enumerate(crop):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (input_shape[1], input_shape[2]), interpolation = cv2.INTER_CUBIC)
        frame = frame.astype(np.float32)
        frame = frame / 255.
        frame = np.expand_dims(frame, 0)

        interpreter.set_tensor(input_details[0]['index'], frame)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        if not reserved[i]:
            if (output[0][0] <= threshold):
                pred.append(0)
            else:
                pred.append(1)
        else:
            pred.append(2)
    
    # try:
    #     img = pre.getImage_masked(pred)
    # except:
    #     img = np.ones((1200, 1600, 3), dtype="uint8")x
    return pre.getImage_masked(pred)

def on_press(key):
    global terminate
    try:
        if key.char == 'x':
            terminate = True
            logging.info("Terminate program")
    except AttributeError:
        pass
    finally:
        return

if __name__ == '__main__':
    connect = False
    while not connect:
        try:
            user = auth.sign_in_with_email_and_password(firebase_config["credential"]["email"], firebase_config["credential"]["password"])
            camConfig_stream = db.child("cam_config").stream(pars.stream_handler, user['idToken'])
            systemConfig_stream = db.child("system_config").stream(pars.config_handler, user['idToken'])
            connect = True
        except:
            logging.info("connection to database failed")
            time.sleep(2)
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while not pars.config_ready() and not terminate:
        logging.info("wait for config...\n")
        time.sleep(2)
    
    json_raw = pars.get_json_raw()
    logging.info(f"System config is >>> start_time: {pars.get_start_time()}, end_time: {pars.get_end_time()}, threshold: {pars.get_threshold()}, update_rate: {pars.get_update_rate()}, cam_timeout: {pars.get_cam_timeout()}")
    last = time.time()

    logging.info(f"camera list:\n{pars.get_cam_name()}")
    cam_index = int(input())

    start = time.time()
    frame = np.ones((1200, 1600, 3), dtype="uint8")
    while not terminate:
        cv2.imshow('monitor', frame)
        cv2.waitKey(20)
        # if (time.time() - start > pars.get_update_rate()):
        if (time.time() - start > 1):
            start = time.time()
            frame = capture(pars.get_url()[cam_index], pars.get_masking()[cam_index], pars.get_slot_reserved()[cam_index], pars.get_cam_timeout(), pars.get_threshold())

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
    
    cv2.destroyAllWindows()
    camConfig_stream.close()
    systemConfig_stream.close()