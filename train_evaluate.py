#import python modules
import os
import glob
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


def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        # Hook to dump intermediate model during training
        modelDumpHook = glob.glob(os.path.join(os.getcwd(), "dumpModel.txt"))
        if modelDumpHook:
            modelName = "model_epoch-{}_batchIdx-{}".format(epoch, batch_idx)
            torch.save(model.state_dict(), modelName)
            os.remove(os.path.join(os.getcwd(), "dumpModel.txt"))

        # Hook to pause exexution flow to debug things
        pauseTrainHook = glob.glob(os.path.join(os.getcwd(), "hook.txt"))
        if pauseTrainHook:
            import pdb
            pdb.set_trace()
            os.remove(os.path.join(os.getcwd(), "hook.txt"))


        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
    train_loss = float(np.mean(losses))
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))
    return train_loss

def test(model, device, test_loader, criterion):
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0
    
    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)

            # Predict for data by doing forward pass
            output = model(data)
            
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())

    test_loss = float(np.mean(losses))
    print('Test set: Average loss: {:.4f}'.format(test_loss))
    return test_loss

#split the input image and the mask 
def split_image(image):
    image = np.array(image)
    Input, mask = image[:, :256, :], image[:, 256:, :] # split the 512x256 into 2 images of size 256x256
    return Input, mask

def run_main(FLAGS):
    #check if GPU is available
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print("GPU is available :" ,device)

    #creating an color array
    num_items = 1000
    np.random.seed(24)
    color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)

    #Kmeans clustering for colour coding the classes
    num_classes = 10
    label_model = KMeans(n_clusters=num_classes)
    label_model.fit(color_array)

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

    # initialise the UNET model
    model = Unet(num_classes=num_classes).to(device)

    # initialise the loss and the optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # train the model
    training_loss = []
    testing_loss = []
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, 
                        criterion, epoch, FLAGS.batch_size)
        test_loss = test(model, device, test_loader, criterion)

        training_loss.append(train_loss)
        testing_loss.append(test_loss)
        
        torch.save(model.state_dict(), "model_epoch_{}".format(epoch))
            
    print("Training and evaluation finished")\

    # Plot the loss
    plt.figure(figsize=(10, 5))
    plt.title("Train Vs Test loss")
    plt.plot(training_loss, label='train loss', linewidth=2, color="b")
    plt.plot(testing_loss, label='test loss', linewidth=2, color="r")
    plt.legend()
    plt.savefig('loss.jpg',  dpi=900)


if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--datasetHome',
                        default=".\\cityscapes_data\\",
                        help='Dataset home folder')
    parser.add_argument('--learning_rate',
                        type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=1,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=2,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    run_main(FLAGS)








