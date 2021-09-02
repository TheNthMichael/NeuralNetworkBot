from os import stat
from dataCollector import DataCollector
import cv2
import ctypes
import pynput
import time
import numpy as np
import dataEncoder
from mss import mss
from PIL import Image
from pynput.keyboard import Key, Controller as KeyController, Listener as KeyListener
from pynput.mouse import Listener as MouseListener
import nnmodel
import stateManager

def on_press_handler(key):
    stateManager.try_add_key_pressed(key)
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
          
    except AttributeError:
        print('special key {0} pressed'.format(key))
    return stateManager.is_not_exiting

def on_release_handler(key):
    if key == Key.f3:
        stateManager.is_recording = not stateManager.is_recording
        return stateManager.is_not_exiting
    stateManager.try_remove_key_pressed(key)
    print('{0} released'.format(key))
    return stateManager.is_not_exiting

def play(model_path: str):
    model = nnmodel.load_model(model_path)
    keyboard = KeyController()
    sct = mss()
    resize_tuple = (1920 //  stateManager.screen_cap_scale, 1080 //  stateManager.screen_cap_scale)
    resolution = resize_tuple[0] * resize_tuple[1]
    # used to record the time when we processed last frame
    prev_frame_time = 0

    fucking_let_go_of_my_keys = []

    # used to record the time at which we processed current frame
    new_frame_time = 0
    sampleNum = 0
    prev_mousex = 1920 // 2
    prev_mousey = 1080 // 2
    with KeyListener(on_press = on_press_handler,
            on_release = on_release_handler) as key_listener:
        while stateManager.is_not_exiting:
            sct.get_pixels(stateManager.monitor_region)
            img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, resize_tuple)

            cur_mousex = stateManager.last_mousex
            cur_mousey = stateManager.last_mousey

            mousedx = cur_mousex - prev_mousex
            mousedy = cur_mousey - prev_mousey

            prev_mousex = cur_mousex
            prev_mousey = cur_mousey

            this_frame_key_count = stateManager.update_keys_frame_count()
            
            
            #img = cv2.Canny(img, threshold1=0, threshold2=200)
            if stateManager.is_recording:
                flat = img.reshape([resolution, 1])
                flat = np.append(flat, this_frame_key_count)
                flat = flat.reshape([flat.shape[0], 1])
                output = model.feedforward(flat)
                mouseOutput = output[-2:]
                keys = output[:-2]
                keys = [1 if x > 0.75 else 0 for x in keys]

                fucking_let_go_of_my_keys = keys[:]
                
                printkey = f"Keys: {stateManager.get_keys_pressed(keys)}"

                #mouseOutput[0], mouseOutput[1] = stateManager.encoder.map_to_real(mouseOutput[0], mouseOutput[1])
                mouseOutput = model.deregularize_mouse(mouseOutput)
                printmouse = f"Mouse: ({int(mouseOutput[0])}, {int(mouseOutput[1])})"
                for i in range(len(keys)):
                    key = dataEncoder.CODE_TO_KEY_MAP[i]
                    if keys[i] == 1:
                        keyboard.press(key)
                    else:
                        keyboard.release(key)
                print(f"Moving Mouse: ({int(mouseOutput[0])}, {int(mouseOutput[1])})")
                x = int(mouseOutput[0])
                y = int(mouseOutput[1])
                #ctypes.windll.user32.mouse_event(0x01, x, y, 0, 0)

                # converting the fps to string so that we can display it on frame
                # by using putText function

                # time when we finish processing for this frame
                new_frame_time = time.time()

                # fps will be number of frame processed in given time frame
                # since their will be most of time error of 0.001 second
                # we will be subtracting it to get more accurate result
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time

                # converting the fps into integer
                fps = int(fps)

                fps = f"fps: {fps}"

                # putting the FPS count on the frame
                cv2.putText(img, fps, (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, printkey, (7, 55), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, printmouse, (7, 85), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
            
            else:
                cv2.putText(img, "Not Recording", (7, 25), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
                for i in range(len(fucking_let_go_of_my_keys)):
                    key = dataEncoder.CODE_TO_KEY_MAP[i]
                    if fucking_let_go_of_my_keys[i] == 1:
                        keyboard.release(key)
                        fucking_let_go_of_my_keys[i] = 0

            cv2.imshow('ScreenCap', img)
            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.destroyAllWindows()
                stateManager.is_not_exiting = False
                break
        key_listener.join()