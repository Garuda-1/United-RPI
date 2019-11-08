import torch

class FaceNet(torch.nn.Module):
    def __init__(self, size):
        super(FaceNet, self).__init__()
        kernel_size = size
        padding = (kernel_size // 2)
        self.con1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernel_size, padding=padding)
        self.ac1 = torch.nn.PReLU()
        self.con2 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding)
        self.ac2 = torch.nn.PReLU()
        self.con3 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding)
        self.ac3 = torch.nn.PReLU()
        self.con4 = torch.nn.Conv2d(in_channels=8, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.ac4 = torch.nn.PReLU()
        self.con5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.ac5 = torch.nn.PReLU()
        self.con6 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.ac6 = torch.nn.PReLU()
        self.con7 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding)
        self.ac7 = torch.nn.PReLU()
        self.con8 = torch.nn.Conv2d(in_channels=64, out_channels=8, kernel_size=kernel_size, padding=padding)
        self.ac8 = torch.nn.PReLU()
        self.con9 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding)
        self.ac9 = torch.nn.PReLU()
        self.con10 = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=kernel_size, padding=padding)
        self.ac10 = torch.nn.PReLU()
        self.con11 = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=kernel_size, padding=padding)
        self.ac11 = torch.nn.Sigmoid()


        self.bn_2d_0 = torch.nn.BatchNorm2d(1)
        self.bn_2d_1 = torch.nn.BatchNorm2d(8)
        self.bn_2d_2 = torch.nn.BatchNorm2d(8)
        self.bn_2d_3 = torch.nn.BatchNorm2d(8)
        self.bn_2d_4 = torch.nn.BatchNorm2d(64)
        self.bn_2d_5 = torch.nn.BatchNorm2d(64)
        self.bn_2d_6 = torch.nn.BatchNorm2d(64)
        self.bn_2d_7 = torch.nn.BatchNorm2d(64)
        self.bn_2d_8 = torch.nn.BatchNorm2d(8)
        self.bn_2d_9 = torch.nn.BatchNorm2d(8)
        self.bn_2d_10 = torch.nn.BatchNorm2d(8)

    def forward(self, x):
        x = self.bn_2d_0(x)
        x = self.ac1(self.con1(x))
        x = self.bn_2d_1(x)
        x = self.ac2(self.con2(x))
        x = self.bn_2d_2(x)
        x = self.ac3(self.con3(x))
        x = self.bn_2d_3(x)
        x = self.ac4(self.con4(x))
        x = self.bn_2d_4(x)
        x = self.ac5(self.con5(x))
        x = self.bn_2d_5(x)
        x = self.ac6(self.con6(x))
        x = self.bn_2d_6(x)
        x = self.ac7(self.con7(x))
        x = self.bn_2d_7(x)
        x = self.ac8(self.con8(x))
        x = self.bn_2d_8(x)
        x = self.ac9(self.con9(x))
        x = self.bn_2d_9(x)
        x = self.ac10(self.con10(x))
        x = self.bn_2d_10(x)
        x = self.ac11(self.con11(x))
        return x;


face_net = FaceNet(25)
face_net.load_state_dict(torch.load('model', map_location='cpu'))
face_net.eval()
face_net.requires_grad = False

