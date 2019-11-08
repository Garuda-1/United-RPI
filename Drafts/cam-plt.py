import time
import picamera
import numpy as np
import torch
import matplotlib.pyplot as plt

def getImage(h, w, awb_mode='auto'):
    camera.awb_mode = awb_mode
    camera.resolution = (w, h)
    camera.framerate = 24
    time.sleep(2)
    npOutput = np.empty((h, w, 3), dtype=np.uint8)
    camera.capture(npOutput, 'rgb')
    return torch.from_numpy(npOutput)

with picamera.PiCamera() as camera:
    camera.start_preview()
    for mode in picamera.PiCamera().AWB_MODES:
        plt.imshow(getImage(240, 320, mode))
        plt.show()
    camera.stop_preview()
