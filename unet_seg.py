#import Python modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim


# defining the UNet Network Architecture
class Unet(nn.Module):
    
    def __init__(self, num_classes):
        super(Unet, self).__init__()
        self.num_classes = num_classes

        #encoder architecture components
        self.encoder_L1 = self.conv_layer(in_channels=3, out_channels=64)
        self.encoder_L2 = self.conv_layer(in_channels=64, out_channels=128)
        self.encoder_L3 = self.conv_layer(in_channels=128, out_channels=256)
        self.encoder_L4 = self.conv_layer(in_channels=256, out_channels=512)
        
        self.pooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pooling_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        #intermediate representation
        self.middle = self.conv_layer(in_channels=512, out_channels=1024)

        #decoder architecture components
        self.decoder_L1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_L2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_L3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_L4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.pooling_5 = self.conv_layer(in_channels=1024, out_channels=512)
        self.pooling_6 = self.conv_layer(in_channels=512, out_channels=256)
        self.pooling_7 = self.conv_layer(in_channels=256, out_channels=128)
        self.pooling_8 = self.conv_layer(in_channels=128, out_channels=64)

        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
    
    #defining a convolutional layer
    def conv_layer(self, in_channels, out_channels):
        layer = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return layer
    
    #defining a forward pass function
    def forward(self, X):
        encoder_L1_out = self.encoder_L1(X) # [-1, 64, 256, 256]
        pooling_1_out = self.pooling_1(encoder_L1_out) # [-1, 64, 128, 128]
        encoder_L2_out = self.encoder_L2(pooling_1_out) # [-1, 128, 128, 128]
        pooling_2_out = self.pooling_2(encoder_L2_out) # [-1, 128, 64, 64]
        encoder_L3_out = self.encoder_L3(pooling_2_out) # [-1, 256, 64, 64]
        pooling_3_out = self.pooling_3(encoder_L3_out) # [-1, 256, 32, 32]
        encoder_L4_out = self.encoder_L4(pooling_3_out) # [-1, 512, 32, 32]
        pooling_4_out = self.pooling_4(encoder_L4_out) # [-1, 512, 16, 16]
        middle_out = self.middle(pooling_4_out) # [-1, 1024, 16, 16]
        decoder_L1_out = self.decoder_L1(middle_out) # [-1, 512, 32, 32]
        pooling_5_out = self.pooling_5(torch.cat((decoder_L1_out, encoder_L4_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        decoder_L2_out = self.decoder_L2(pooling_5_out) # [-1, 256, 64, 64]
        pooling_6_out = self.pooling_6(torch.cat((decoder_L2_out, encoder_L3_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        decoder_L3_out = self.decoder_L3(pooling_6_out) # [-1, 128, 128, 128]
        pooling_7_out = self.pooling_7(torch.cat((decoder_L3_out, encoder_L2_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        decoder_L4_out = self.decoder_L4(pooling_7_out) # [-1, 64, 256, 256]
        pooling_8_out = self.pooling_8(torch.cat((decoder_L4_out, encoder_L1_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(pooling_8_out) # [-1, num_classes, 256, 256]
        return output_out   #final segmentation map
