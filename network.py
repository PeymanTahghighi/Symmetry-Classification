#===============================================================
#===============================================================
from copy import deepcopy
from shutil import copy
import torch
import torch.nn as nn
from torchvision import models
#===============================================================
#===============================================================

#---------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super().__init__();
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.net(x);
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, type) -> None:
        super().__init__();

        if type == 'convtrans':
            self.net = nn.ConvTranspose2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=2, padding=1);
        else:
            self.net = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=1)
            )
        
        self.conv_after = nn.Sequential(
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1),
            ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1))
    
    def forward(self, x):
        x = self.net(x);
        x = self.conv_after(x);
        return x;
#---------------------------------------------------------------

#---------------------------------------------------------------
class Upblock(nn.Module):
    def __init__(self, in_features, out_features, concat_features = None) -> None:
        super().__init__();
        if concat_features == None:
            concat_features = out_features*2;

        self.upsample = Upsample(in_features, out_features, 4, 'convtrans');
        self.conv1 = ConvBlock(in_channels=concat_features, out_channels=out_features, kernel_size=3, stride=1);
        self.conv2 = ConvBlock(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1);

    def forward(self, x1, x2):
        x1 = self.upsample(x1);
        ct = torch.cat([x1,x2], dim=1);
        ct = self.conv1(ct);
        out = self.conv2(ct);
        return out;
#---------------------------------------------------------------

#---------------------------------------------------------------
class Unet(nn.Module):
    def __init__(self, num_classes = None) -> None:
        super().__init__();
        resnet = models.resnet50(pretrained= True);
        self.input_blocks = nn.Sequential(*list(resnet.children()))[:3];
        self.input_pool = list(resnet.children())[3];
        resnet_down_blocks = [];
        for btlnck in list(resnet.children()):
            if isinstance(btlnck, nn.Sequential):
                resnet_down_blocks.append(btlnck);
        
        self.down_blocks = nn.Sequential(*resnet_down_blocks);

        self.bottle_neck = nn.Sequential(
            ConvBlock(2048, 2048, 3, 1),
            ConvBlock(2048, 2048, 3, 1)
        );

        self.up_1 = Upblock(2048,1024);
        self.up_2 = Upblock(1024,512);
        self.up_3 = Upblock(512,256);
        self.up_4 = Upblock(256, 128, 128+64)
        self.up_5 = Upblock(128, 64, 64+3);
        
        if num_classes != None:
            self.final = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0);
        

        self.__initial_weights = deepcopy(self.state_dict());
        
    def forward(self, inp):
        d_1 = self.input_blocks(inp);

        d_1_pool = self.input_pool(d_1);
        d_2 = self.down_blocks[0](d_1_pool);
        d_3 = self.down_blocks[1](d_2);
        d_4 = self.down_blocks[2](d_3);
        d_5 = self.down_blocks[3](d_4);

        d_5 = self.bottle_neck(d_5);

        u_1 = self.up_1(d_5, d_4);
        u_2 = self.up_2(u_1, d_3);
        u_3 = self.up_3(u_2, d_2);
        u_4 = self.up_4(u_3, d_1);
        u_5 = self.up_5(u_4, inp);

        out = self.final(u_5);

        return out;
    

    def reset_weights(self):
        self.load_state_dict(self.__initial_weights);
#---------------------------------------------------------------

#---------------------------------------------------------------
def test():

    sample = torch.rand((1,3,512,512));
    net = Unet(5);
    out,_ = net(sample);
    print(out.size());
#---------------------------------------------------------------

#---------------------------------------------------------------
if __name__ == "__main__":
    test();
#---------------------------------------------------------------