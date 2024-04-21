import time
import sys
import cv2
import os
import numpy as np
from PIL import Image
from math import *
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import models, transforms
from torch.utils.data import Dataset

if not os.path.exists('/content/ibug_300W_large_face_landmark_dataset'):
    !wget http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz
    !tar -xvzf 'ibug_300W_large_face_landmark_dataset.tar.gz'
    !rm -r 'ibug_300W_large_face_landmark_dataset.tar.gz'

class Transforms():
    def __init__(self):
        pass

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
      left = int(crops['left'])
      top = int(crops['top'])
      width = int(crops['width'])
      height = int(crops['height'])

      image = TF.crop(image, top, left, height, width)

      if landmarks is not None and len(landmarks) > 0:
          img_shape = np.array(image).shape
          landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
          landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])

      return image, landmarks


    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        # image, landmarks = self.rotate(image, landmarks, angle=10)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks
    

class FaceLandmarksDataset(Dataset):
    
    def __init__(self, transform=None):

        tree = ET.parse('ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = 'ibug_300W_large_face_landmark_dataset'

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        # Calculate interocular distance
        self.interocular_distances = []
        for landmark in self.landmarks:
            left_eye = landmark[36]  # Assuming landmarks are in iBUG annotation format
            right_eye = landmark[45]
            interocular_distance = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
            self.interocular_distances.append(interocular_distance)


        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]
        interocular_distance = self.interocular_distances[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks, interocular_distance

dataset = FaceLandmarksDataset(Transforms())

# split the dataset into validation and test sets
len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set

train_dataset, valid_dataset, = torch.utils.data.random_split(dataset , [len_train_set, len_valid_set])

# shuffle and batch the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=True, num_workers=2)

images, landmarks, interocular_distance = next(iter(train_loader))

class Network(nn.Module):
    def __init__(self,num_classes=136):
        super().__init__()
        self.model_name='resnet18'
        self.model=models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x=self.model(x)
        return x
    

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

    sys.stdout.flush()

def calculate_nme(predictions, targets, interocular_distance):
    euclidean_distances = np.linalg.norm(predictions - targets, axis=-1)
    normalized_distances = euclidean_distances / interocular_distance
    mean_nme = np.mean(np.float64(normalized_distances))
    return mean_nme

torch.autograd.set_detect_anomaly(True)
network = Network()
# network.cuda()
network.to('cuda')

criterion = nn.MSELoss()
optimizer = optim.Adam(network.parameters(), lr=0.0001)

loss_min = np.inf
num_epochs = 10

train_nme_values = []
valid_nme_values = []

start_time = time.time()

for epoch in range(1,num_epochs+1):

    loss_train = 0
    loss_valid = 0
    running_loss = 0

    network.train()
    train_loader_iter = iter(train_loader)
    valid_loader_iter = iter(valid_loader)

    epoch_train_nme_values = []


    for step in range(1, len(train_loader_iter) + 1):

        images, landmarks, interocular_distance = next(train_loader_iter)

        # images = images.cuda()
        images = images.to('cuda')
        # landmarks = landmarks.view(landmarks.size(0),-1).cuda()
        landmarks = landmarks.view(landmarks.size(0),-1).to('cuda')

        predictions = network(images)

        nme_train_step = calculate_nme(predictions.detach().cpu().numpy(), landmarks.cpu().numpy(), interocular_distance)
        epoch_train_nme_values.append(nme_train_step)


        print_overwrite(step, len(train_loader), running_loss, 'train')
        print(f" NME: {nme_train_step:.4f}")
        # clear all the gradients before calculating them
        optimizer.zero_grad()

        # find the loss for the current step
        loss_train_step = criterion(predictions, landmarks)

        # calculate the gradients
        loss_train_step.backward()

        # update the parameters
        optimizer.step()

        loss_train += loss_train_step.item()
        running_loss = loss_train/step

        print_overwrite(step, len(train_loader), running_loss, 'train')

    mean_nme_train_epoch = np.mean(epoch_train_nme_values)
    train_nme_values.append(mean_nme_train_epoch)
    print(train_nme_values)
    network.eval()
    with torch.no_grad():

        epoch_valid_nme_values = []
        for step, (images, landmarks, interocular_distance) in enumerate(valid_loader, 1):


            # images = images.cuda()
            images = images.to('cuda')
            # landmarks = landmarks.view(landmarks.size(0),-1).cuda()
            landmarks = landmarks.view(landmarks.size(0),-1).to('cuda')

            predictions = network(images)
            # calculate metrics
            nme_valid_step = calculate_nme(predictions.detach().cpu().numpy(), landmarks.cpu().numpy(), interocular_distance)
            epoch_valid_nme_values.append(nme_valid_step)


            # Print or log the metrics
            print_overwrite(step, len(valid_loader), running_loss, 'valid')
            print(f" NME: {nme_valid_step:.4f}")

            # find the loss for the current step
            loss_valid_step = criterion(predictions, landmarks)

            loss_valid += loss_valid_step.item()
            running_loss = loss_valid/step

            print_overwrite(step, len(valid_loader), running_loss, 'valid')

    mean_nme_valid_epoch = np.mean(epoch_valid_nme_values)
    valid_nme_values.append(mean_nme_valid_epoch)


    loss_train /= len(train_loader)
    loss_valid /= len(valid_loader)

    print('\n--------------------------------------------------')
    print('Epoch: {}  Train Loss: {:.4f}  Valid Loss: {:.4f}'.format(epoch, loss_train, loss_valid))
    print('--------------------------------------------------')

    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(network.state_dict(), '/content/drive/MyDrive/CV/Model/face_landmarks.pth')
        print("\nMinimum Validation Loss of {:.4f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
        print('Model Saved\n')

print('Training Complete')
print("Total Elapsed Time : {} s".format(time.time()-start_time))