import torch 
import torch.nn as nn 


class Discrimiator(nn.Module):
    def __init__(self,  num_channels, maps):
        super(Discrimiator, self).__init__()
        print(maps)
        self.net = nn.Sequential(
                # INPUT : N x C x 64 x 64
                nn.Conv2d(in_channels = num_channels, out_channels=maps, kernel_size=4, stride=2, padding=1), #N x C x 32 x 32
                self._block(in_channels=maps, out_channels=maps*2, kernel_size=4 , stride=2, padding=1),      # N x C x 16 x 16
                self._block(in_channels=maps*2, out_channels=maps*4, kernel_size=4 , stride=2, padding=1),    # N x C x 8 x 8
                self._block(in_channels=maps*4, out_channels=maps*8, kernel_size=4 , stride=2, padding=1),    # N x C x 4 x 4 
                nn.Conv2d(in_channels=maps*8, out_channels=1, kernel_size=4 , stride=2, padding=0),           # N x 1 x 1 x 1     
                nn.Sigmoid()
        )

    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x)



class Generator(nn.Module):
    def __init__(self, noise_channels ,img_channels, maps):
        super(Generator, self).__init__()
        print(maps)
        self.net = nn.Sequential(
            # input  = N x 100 x 1 x 1 
                self._block(in_channels=noise_channels, out_channels=maps*16, kernel_size=4 , stride=1, padding=0),    # N x C x 4 x 4
                self._block(in_channels=maps*16, out_channels=maps*8, kernel_size=4 , stride=2, padding=1),    # N x C x 8 x 8
                self._block(in_channels=maps*8, out_channels=maps*4, kernel_size=4 , stride=2, padding=1),    # N x C x 16 x 16
                self._block(in_channels=maps*4, out_channels=maps*2, kernel_size=4 , stride=2, padding=1),    # N x C x 32 x 32

                nn.ConvTranspose2d(in_channels = maps*2, out_channels=img_channels, kernel_size=4, stride=2, padding=1), # N x C x 64 x 64
                nn.Sigmoid()
        )

    def _block(self,in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)



def initialize_wieghts(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.00, 0.02)


def test_discremenator():
    x = torch.rand(32, 3, 64, 64)
    disc = Discrimiator(num_channels=3, maps = 32)
    print(disc(x).shape)


def test_generator():
    x = torch.rand(32, 100,1, 1)
    disc = Generator(noise_channels=100, img_channels = 3, maps = 32)
    print(disc(x).shape)


if __name__ == '__main__':
    test_discremenator()
    test_generator()