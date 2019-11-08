import time
import picamera
import numpy as np
import torch 



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
        transforms.Resize((256, 256))
        transforms.ToTensor()
    ])

    capture_out = test_transforms(Image.fromarray(source))

