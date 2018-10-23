import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

ap= argparse.ArgumentParser(description='Train.py')
ap.add_argument('data_dir', nargs='*', action="store",default="./flowers/")
ap.add_argument('--gpu',dest="gpu",action="store",default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest="dropout", action="store", default=0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

arch = {"vgg16":25088,
       "densenet121":1024,
       "alexnet":9216}

    
data_dir = where
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
valid_transforms =transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
vloader = torch.utils.data.DataLoader(validation_data, batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size =  20, shuffle = True)
    
def nn_setup(structure = 'densenet121', dropout = 0.5, hidden_layer1 = 120, lr = 0.001, power = 'gpu'):
    if structure == 'vgg16':
        model = models.vgg16(pretrained = True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained = True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else: 
        print("I am Sorry but {} is not a valid model. Did you mean vgg16, densenet121, or alexnet ? ".format(structure))
    
    for param in model.parameters():
        param.requires_grad = False 
        
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('inputs', nn.Linear(arch[structure],hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(90,80)),
            ('relu3', nn.ReLU()),
            ('hidden_layer3', nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))]))
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        
        if torch.cuda.is_available() and power=='gpu':
            model.cuda()
            
        return model, criterion, optimizer
    
def train_network(model, criterion, optimizer, epochs =3, print_every = 20, power = 'gpu'):
    steps = 0
    running_loss = 0
    
    print("----------Training is starting-------- ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps +=1
            if torch.cuda.is_available() and power == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
            optimizer.zero_grad()
            
            #Forward and Backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy =0
                
                for ii, (inputs2, labels2) in enumerate(vloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0'), labels2.to('cuda:0')
                        model.to('cuda:0')
                        
                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs, labels2)
                        ps= torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()
                        
                vlost= vlost / len(vloader)
                accuracy = accuracy / len(vloader)
                
                print("Epoch: {}/{}...".format(e+1,epochs),
                     "Loss: {:.4f}".format(running_loss / print_every),
                     "Validation Lost {:.4f}".format(vlost),
                     "Accuracy: {:.4f}".format(accuracy))
                
                running_loss = 0
                
    print("*******Finished Training*******")
    print("Dear User I the ulitmate NN machine trained your model. It required")
    print("***Epochs: {}***".format(epochs))
    print("***Steps: {}***".format(steps))
    print("That's a lot of steps")
    
def save_checkpoint(path='checkpoint.pth', structure = 'densenet121', hidden_layer1 = 120, dropout = 0.5, lr = 0.001, epochs = 12):
    #This function saves the model at a specified by the user path
    
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure': structure,
               'hidden_layer1':hidden_layer1,
               'dropout':dropout,
               'lr': lr,
               'nb_of_epochs': epochs,
               'state_dict':model.state_dict(),
               'class_to_idx': model.class_to_idx},
              path)

    

model, optimizer, criterion = nn_setup(structure, dropout, hidden_layer1, lr, power)

train_network(model, optimizer, criterion,epochs, 20, power)

save_checkpoint(path, structure, hidden_layer1, dropout, lr)

print("All Model is Trained")