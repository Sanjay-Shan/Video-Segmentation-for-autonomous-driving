#import python modules

import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from unet_seg import Unet
from Dataset import Cityscape



device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

#Load the trained model
num_classes = 10
model_path = "model_epoch_20" 
num_items = 1000
np.random.seed(24)
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
label_model = KMeans(n_clusters=num_classes)
label_model.fit(color_array)
model_ = Unet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_path))

#Load the test set for inference
data_dir = os.path.join("cityscapes_data")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_batch_size = 2
train_dataset = Cityscape(train_dir, label_model)
train_loader = DataLoader(train_dataset, batch_size=test_batch_size, shuffle=True)
val_dataset = Cityscape(val_dir, label_model)
val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)

# infer on the test set using an iterator
X, Y = next(iter(train_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
Y_pred = torch.argmax(Y_pred, dim=1)

inverse_transform = transforms.Compose([transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))])
fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5)) #create a grid for visualization

#Segmentation output visualization
def generatePlot(unique, cmap):
    for i in range(test_batch_size):
        landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
        label_class = Y[i].cpu().detach().numpy()
        label_class_predicted = Y_pred[i].cpu().detach().numpy()
        # label_class = Y[i].cpu().detach()
        
        axes[i, 0].imshow(landscape, cmap=cmap)
        axes[i, 0].set_title("Input Image")
        axes[i, 1].imshow(label_class, cmap=cmap)
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 2].imshow(label_class_predicted, cmap=cmap)
        axes[i, 2].set_title("Segmented output - Predicted")

    plt.savefig("result_{}.jpg".format(unique))

import pdb
pdb.set_trace()
generatePlot("final", "Set2")
# infer on the test set using an iterator
X, Y = next(iter(val_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
Y_pred = torch.argmax(Y_pred, dim=1)

inverse_transform = transforms.Compose([transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))])
fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5)) #create a grid for visualization

#Segmentation output visualization
for i in range(test_batch_size):
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Input Image")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Ground Truth Mask")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Segmented output - Predicted")

plt.savefig("val_output.jpg")