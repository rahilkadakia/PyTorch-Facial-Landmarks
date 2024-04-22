
import matplotlib.pyplot as plt
import io
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

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

class Transforms():
    def __init__(self):
        pass

    def resize(self, image, img_size):
        image = TF.resize(image, img_size)
        return image

    def color_jitter(self, image):
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image

def process_image(model):
    image = Image.open("168907806_1.jpg")
    image = image.convert("L")
    transforms = Transforms()
    image = transforms.resize(image, (224, 224))
    image = transforms.color_jitter(image)
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5]).unsqueeze(0)

    predictions = (model(image).cpu() + 0.55) * 190
    predictions = predictions.view(-1,68,2)
    plt.figure(figsize=(5,5))
    plt.imshow(image.numpy().squeeze(), cmap='gray')
    plt.scatter(predictions[0,:,0].detach().numpy(), predictions[0,:,1].detach().numpy(), c='r', s=5)
    # plt.show()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    return buffer

    # return image

# model = torch.load("Models/model.pt", map_location=torch.device('cpu'))
# print ("###### MODEL LOADED ######")
# model.load_state_dict(torch.load("Models/model.pth", map_location=torch.device('cpu'))) # load model only once, when server is started
# print ("###### MODEL STATE DICT LOADED ######")
# model.eval()
# process_image(model)