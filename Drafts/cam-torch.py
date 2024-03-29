import time
import picamera
import numpy as np
import torch 

with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.framerate = 24
    time.sleep(2)
    npOutput = np.empty((240, 320, 3), dtype=np.uint8)
    camera.capture(npOutput, 'rgb')

    torchOutput = torch.from_numpy(npOutput)
    print(torchOutput)
