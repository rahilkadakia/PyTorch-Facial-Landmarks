import matplotlib
import matplotlib.pyplot as plt
import io
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

matplotlib.use('Agg')
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

class SingleImageDataset(Dataset):
    def __init__(self, img, transform=None):
        self.img = img
        self.transform = transform

    def __len__(self):
        return 1  # Since we have only one image

    def __getitem__(self, idx):
        if self.transform:
            img = self.transform(self.img)
        else:
            img = transforms.ToTensor()(self.img)
        return img

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
    
    def to_tensor(self, image):
        image = TF.to_tensor(image)
        return image

    def normalize(self, image):
        image = TF.normalize(image, [0.5], [0.5])
        return image

def process_image(model, image):
    image = Image.open(io.BytesIO(image))
    image = image.convert("L")
    
    custom_transforms = Transforms()
    transform = transforms.Compose([
        transforms.Lambda(lambda x: custom_transforms.resize(x, (224, 224))),  
        transforms.Lambda(lambda x: custom_transforms.color_jitter(x)),
        transforms.Lambda(lambda x: custom_transforms.to_tensor(x)),
        transforms.Lambda(lambda x: custom_transforms.normalize(x))
    ])
    
    single_image_dataset = SingleImageDataset(image, transform=transform)
    data_loader = DataLoader(single_image_dataset, batch_size=1, shuffle=False)

    image = next(iter(data_loader))

    predictions = (model(image).cpu() + 0.5) * 224
    predictions = predictions.view(-1,68,2)
    plt.figure(figsize=(5,5))

    plt.imshow(image[0].cpu().numpy().transpose(1,2,0).squeeze(), cmap='gray')
    plt.scatter(predictions[0,:,0].detach().numpy(), predictions[0,:,1].detach().numpy(), c='r', s=5)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.show()
    return buffer