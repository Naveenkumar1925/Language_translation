import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import cv2
import sys
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

class MRI(Dataset):
    def __init__(self):
        
        tumor = []
        healthy = []
        
        for f in glob.iglob("./data/brain_tumor_dataset/yes1/*.jpg"):
            img = cv2.imread(f)
            img = cv2.resize(img,(128,128))
            b, g, r = cv2.split(img)
            img = cv2.merge([r,g,b])
            img = img.reshape((img.shape[2],img.shape[0],img.shape[1])) 
            tumor.append(img)

        for f in glob.iglob("./data/brain_tumor_dataset/no/*.jpg"):
            img = cv2.imread(f)
            img = cv2.resize(img,(128,128)) 
            b, g, r = cv2.split(img)
            img = cv2.merge([r,g,b])
            img = img.reshape((img.shape[2],img.shape[0],img.shape[1]))
            healthy.append(img)

        
        tumor = np.array(tumor,dtype=np.float32)
        healthy = np.array(healthy,dtype=np.float32)
        
        
        tumor_label = np.ones(tumor.shape[0], dtype=np.float32)
        healthy_label = np.zeros(healthy.shape[0], dtype=np.float32)
        
        
        self.images = np.concatenate((tumor, healthy), axis=0)
        self.labels = np.concatenate((tumor_label, healthy_label))

        self.tumor_len = len(tumor) 
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        
        sample = {'image': self.images[index], 'label':self.labels[index]}
        
        return sample
    
    def normalize(self):
        self.images = self.images/255.0

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5))
        
        self.fc_model = nn.Sequential(
        nn.Linear(in_features=256, out_features=120),
        nn.Tanh(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=1))
        
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.sigmoid(x)
        
        return x

mri_dataset = MRI()
mri_dataset.normalize()

device = torch.device('cuda')


eta = 0.00001
EPOCH = 10000

dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=True)
model = CNN().to(device)

def train1():
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    """
    if os.path.exists('./model_train_weights2.pth' and './optimizer_train_weights2.pth'):

        model.load_state_dict('./model_train_weights.pth')
        optimizer.load_state_dict('./optimizer_train_weights.pth')"""
    for epoch in range(1, EPOCH):
        losses = []
        for D in dataloader:
            optimizer.zero_grad()
            data = D['image'].to(device)
            label = D['label'].to(device)
            y_hat = model(data)
            
            error = nn.BCELoss() 
            loss = torch.sum(error(y_hat.squeeze(), label))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print('Train Epoch : ',epoch+1,' Loss : ',np.mean(losses))
        if (epoch+1) % 50 == 0:
            #saving the file
            torch.save(model.state_dict(), "model_train_weights3.pth")
            torch.save(optimizer.state_dict(), "optimizer_train_weights3.pth")
            print("Model weights saved to model_train_weights.pth")
train1()

#evaluation 

model.eval()
dataloader = DataLoader(mri_dataset, batch_size=32, shuffle=False)
outputs=[]
y_true = []
with torch.no_grad():
    for D in dataloader:
        image =  D['image'].to(device)
        label = D['label'].to(device)
        
        y_hat = model(image)
        
        outputs.append(y_hat.cpu().detach().numpy())
        y_true.append(label.cpu().detach().numpy())
        
outputs = np.concatenate( outputs, axis=0 )
y_true = np.concatenate( y_true, axis=0 )

def threshold(scores,threshold=0.50, minimum=0, maximum = 1.0):
    x = np.array(list(scores))
    x[x >= threshold] = maximum
    x[x < threshold] = minimum
    return x

print("Accuracy Score ; ",accuracy_score(y_true, threshold(outputs)))

#Heatmap

cm = confusion_matrix(y_true, threshold(outputs))
plt.figure(figsize=(16,9))

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Tumor','Healthy'])
ax.yaxis.set_ticklabels(['Tumor','Healthy'])
plt.show()

plt.figure(figsize=(16,9))
plt.plot(outputs)
plt.axvline(x=mri_dataset.tumor_len, color='r', linestyle='--')
plt.grid()
plt.show()

#feature map of cnn

print("Model : ",model)

no_of_layers = 0
conv_layers = []

model_children = list(model.children())
print("Model children ; ",model_children)

for child in model_children:
    if type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                no_of_layers += 1
                conv_layers.append(layer)

print("Conventional layers : ",conv_layers)

img = mri_dataset[100]['image']
plt.imshow(img.reshape(128,128,3))
plt.show()

img = torch.from_numpy(img).to(device)
print("Img shape : ",img.shape)

img = img.unsqueeze(0)
print("Img shape unsqueeze : ",img.shape)

results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))
outputs = results

for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer].squeeze()
    print("Layer ",num_layer+1)
    for i, f in enumerate(layer_viz):
        plt.subplot(2, 8, i + 1)
        plt.imshow(f.detach().cpu().numpy())
        plt.axis("off")
    plt.show()
    plt.close()

#over fitting ?

