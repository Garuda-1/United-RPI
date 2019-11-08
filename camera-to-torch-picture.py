import time
import picamera
import numpy as np
import torch
import matplotlib.pyplot as plt

with picamera.PiCamera() as camera:
    camera.resolution = (320, 240)
    camera.framerate = 24
    time.sleep(2)
    npOutput = np.empty((240, 320, 3), dtype=np.uint8)
    camera.capture(npOutput, 'rgb')
    npOutput = np.dot(npOutput[...,:3], [0.2989, 0.5870, 0.1440])

    camera.close()

    torchOutput = torch.from_numpy(npOutput)
    print(torchOutput)

    plt.imshow(torchOutput)
    plt.show()

