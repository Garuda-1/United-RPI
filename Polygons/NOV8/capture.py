import time
import picamera
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from skimage.feature import hog
from PIL import ImageEnhance
from PIL import ImageFilter

with picamera.PiCamera() as camera:
    w = 512
    h = 512

    camera.start_preview()

    camera.resolution = (w, h)
    camera.awb_mode = 'auto'
    camera.framerate = 24
    time.sleep(2)

    source = np.empty((w, h, 3), dtype=np.uint8)
    camera.capture(source, 'rgb')

    camera.stop_preview()

    source = Image.fromarray(source, 'RGB')
    source = transforms.Resize((w,h))(source)
    fd, source = hog(source, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    capture_out = test_transforms(Image.fromarray(source))
    capture_out = capture_out[0]
    capture_out /= capture_out.max()
    capture_out *= 255

    data = np.zeros((capture_out.shape[0], capture_out.shape[1], 3), dtype=np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            t = capture_out[i, j]
            data[i, j] = [t, t, t]

    img = Image.fromarray(data, 'RGB')
    img.save('capture_data.png')
