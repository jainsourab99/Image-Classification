import matplotlib.pyplot as plt
import numpy as np
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
import train
ap=argparse.ArgumentParser(description = 'predict-file')
ap.add_argument('input_img',default = '/home/workspace/aipnd-project/flowers/test/1/image_06752.jpg', nargs ='*', action ="store", type=str)
ap.add_argument('checkpoint', default = '/home/workspace/aipnd-project/checkpoint.pth',nargs= "*", action="store", type= str)
ap.add_argument('--top_k',default =5, dest= "top_k",action="store",type= int)
ap.add_argument('--category_names',dest="category_names",action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store",dest="gpu")

pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint

data_dir = "./flowers"
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                         [0.229,0.224,0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485,0.456,0.406],
                                                               [0.229,0.224,0.225])])

train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform = validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
vloader = torch.utils.data.DataLoader(validation_data,batch_size = 32, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)

checkpoint = torch.load('checkpoint.pth')
structure = checkpoint['structure']
hidden_layer1 = checkpoint['hidden_layer1']
dropout = checkpoint['dropout']
lr = checkpoint['lr']

model,_,_ = train.nn_setup(structure, dropout, hidden_layer1, lr)
   
model.class_to_idx= checkpoint['class_to_idx']
model.load_state_dict(checkpoint['state_dict'])

with open('cat_to_name.json','r') as json_file:
    cat_to_name= json.load(json_file)
    
def process_image(image_path):
    
    img = Image.open(image_path)
    
    make_img_good= transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    tensor_image = make_img_good(img)
    return tensor_image


def predict(image_path, model, topk=5, power = 'gpu'):
    if torch.cuda.is_available() and power == 'gpu':
        model.to('cuda:0')
        
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch= img_torch.float()
    
    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)
            
    probability = F.softmax(output.data, dim=1)
    
    return probability.topk(topk)
            
probabilities = predict(path_image,model,number_of_outputs, power)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

i=0
while i< number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i+=1
    
print("All Predict Done")
