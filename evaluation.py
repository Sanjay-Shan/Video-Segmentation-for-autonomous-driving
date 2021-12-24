#import python modules
import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Dataset import Cityscape
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from unet_seg import Unet
from torchmetrics import IoU


def test(model, data_loader, iou):
    # Set model to eval mode to notify all layers.
    model.eval()
    
    iou_values = []
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(data_loader):
            data, target = sample
            
            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute IOU
            iou_value = iou(output, target)  
            iou_values.append(iou_value)

            print('Batch: {},  IOU: {:.4f}'.format(batch_idx, iou_value))

    mean_iou = float(np.mean(iou_values))
    return mean_iou


def run_main(FLAGS):
    #check if GPU is available
    
    #Load the trained model
    num_classes = 10
    model_path = "model_epoch_21" 
    num_items = 1000
    np.random.seed(24)
    color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
    label_model = KMeans(n_clusters=num_classes)
    label_model.fit(color_array)
    model = Unet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))

    iou = IoU(num_classes=num_classes)

    # Load the dataset from the directory
    data_directory = os.path.join(FLAGS.datasetHome)
    train_directory = os.path.join(data_directory, "train") 
    test_directory = os.path.join(data_directory, "val")
    train_split = os.listdir(train_directory)
    test_split = os.listdir(test_directory)
    print("Size of the Train and Test set : "+ str(len(train_split))+"\t"+str(len(test_split)))
    
    # define the network parameters
    batch_size = FLAGS.batch_size
    epochs = FLAGS.num_epochs
    lr = FLAGS.learning_rate

    # Load train data
    train_dataset = Cityscape(train_directory, label_model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # load test data
    test_dataset = Cityscape(test_directory, label_model)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_iou = test(model, train_loader, iou)
    print('Train set: IOU: {:.4f}'.format(train_iou))
    test_iou = test(model, test_loader, iou)
    print('Test set: IOU: {:.4f}'.format(test_iou))
        
    print("Evaluation finished")\



if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--datasetHome',
                        default="cityscapes_data",
                        help='Dataset home folder')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)








