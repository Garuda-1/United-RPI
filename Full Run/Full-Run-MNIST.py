print('CELL 1')
import torch
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


print('CELL 2')
import torchvision.datasets
MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)


print('CELL 3')
X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels


print('CELL 4')
X_train.dtype, y_train.dtype


print('CELL 5')
X_train = X_train.float()
X_test = X_test.float()


print('CELL 6')
X_train.shape, X_test.shape


print('CELL 7')
y_train.shape, y_test.shape


print('CELL 8')
import matplotlib.pyplot as plt
plt.imshow(X_train[0, :, :])
plt.show()
print(y_train[0])


print('CELL 9')
X_train = X_train.reshape([-1, 28 * 28])
X_test = X_test.reshape([-1, 28 * 28])


print('CELL 10')
class MNISTNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(MNISTNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, n_hidden_neurons)
        self.ac1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, 10) 
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.ac1(x)
        x = self.fc2(x)
        return x
    
mnist_net = MNISTNet(100)


print('CELL 11')
torch.cuda.is_available()


print('CELL 12')
# !nvidia-smi


print('CELL 12')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mnist_net = mnist_net.to(device)
list(mnist_net.parameters())


print('CELL 13')
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mnist_net.parameters(), lr=1.0e-3)


print('CELL 14')
batch_size = 100

test_accuracy_history = []
test_loss_history = []

X_test = X_test.to(device)
y_test = y_test.to(device)

for epoch in range(10000):
    order = np.random.permutation(len(X_train))
    
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        
        batch_indexes = order[start_index:start_index+batch_size]
        
        X_batch = X_train[batch_indexes].to(device)
        y_batch = y_train[batch_indexes].to(device)
        
        preds = mnist_net.forward(X_batch) 
        
        loss_value = loss(preds, y_batch)
        loss_value.backward()
        
        optimizer.step()

    test_preds = mnist_net.forward(X_test)
    test_loss_history.append(loss(test_preds, y_test))
    
    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    test_accuracy_history.append(accuracy)
    print(accuracy)


print('CELL 15')
plt.plot(test_accuracy_history)
plt.plot(test_loss_history)

mnist_net.eval()
torch.set_grad_enabled(False)

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

    resizeTransforms = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    npOutput = np.dot(npOutput[...,:3], [0.2989, 0.5870, 0.1440])

    camera.close()

    torchOutput = torch.from_numpy(npOutput)

    plt.imshow(torchOutput)
    plt.show()

    torchOutput = resizeTransforms(Image.fromarray(npOutput))
    torchOutput = 1 - torchOutput

    print(torchOutput)
    print(torchOutput.shape)

    plt.imshow(torchOutput[0, :, :])
    plt.show()

