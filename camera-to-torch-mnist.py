import time
import picamera
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

with picamera.PiCamera() as camera:
    camera.resolution = (320, 320)
    camera.framerate = 24
    time.sleep(2)
    npOutput = np.empty((320, 320, 3), dtype=np.uint8)
    camera.capture(npOutput, 'rgb')

    # tPIL = transforms.ToPILImage()
    # tTen = transforms.ToTensor()

    resizeTransforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    npOutput = np.dot(npOutput[...,:3], [0.2989, 0.5870, 0.1440])

    camera.close()

    torchOutput = torch.from_numpy(npOutput)

    plt.imshow(torchOutput)
    plt.show()

    image = Image.fromarray(npOutput)
    image.show()
    torchOutput = resizeTransforms(image)
    torchOutput = 1 - torchOutput

    #image = tPIL(torchOutput)
    #plt.imshow(np.asarray(image))
    #plt.show()
    #size = 28, 28
    #image = image.resize(size)
    #torchOutput = tTen(image)

    # torchOutput = tTen(tPIL(torchOutput).resize(28, 28, Image.ANTIALIAS))
    # resizeTransform = transforms.Compose([transforms.Resize((28,28))])
    # torchOutput = torchOutput.resize_((28, 28))

    print(torchOutput)
    print(torchOutput.shape)

    plt.imshow(torchOutput[0, :, :])
    plt.show()

