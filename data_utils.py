import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.filters import gaussian
import numpy as np
from PIL import Image
from scipy.misc import imsave, imread
import torchvision.transforms.functional as TF
import torch
import os
import glob
import random
import re


numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

class GaussianFilter:
    """Apply Gaussian filter"""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):

        sigma_factor = random.uniform(0, self.sigma)
        image=gaussian(TF.to_tensor(image).numpy().transpose(1,2,0), sigma_factor, multichannel=True).transpose(2,0,1)
        image=TF.to_pil_image(torch.from_numpy(image))
        return image

class BasicDataset(Dataset):
    def __init__(self,folder_names, relative_path="/*/*", validation = False, testing=False, patch_size=350):
        self.pos_images = list()
        self.neg_images = list()

        self.test_images = list()

        self.testing=testing

        for folder in folder_names:
            for fold in os.listdir(folder):
                # print(fold)
                if not self.testing:
                    if "all" in fold:
                        self.pos_images += sorted(glob.glob(os.path.join(folder,fold)+relative_path))
                    else:
                        self.neg_images += sorted(glob.glob(os.path.join(folder,fold)+relative_path))
                else:

                    self.test_images += sorted(glob.glob(os.path.join(folder,fold)+relative_path),key=numericalSort)




        if not validation:
            try:
                over_sampled_neg_images = random.sample(self.neg_images, len(self.pos_images)- len(self.neg_images))
                self.neg_images += over_sampled_neg_images

            except:
                over_sampled_neg_images = random.sample(self.neg_images*2, len(self.pos_images)- len(self.neg_images))
                self.neg_images += over_sampled_neg_images
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.CenterCrop(patch_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomApply([transforms.RandomAffine(degrees=0, shear=20)], p=.5),
                                                    transforms.RandomApply([GaussianFilter(.75)], p=.2),
                                                    transforms.RandomRotation(360),
                                                    transforms.ToTensor() ])
        else:
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.CenterCrop(patch_size),
                                                    transforms.ToTensor() ])



        self.num_pos_images = len(self.pos_images)
        self.num_neg_images = len(self.neg_images)
        self.num_test_images = len(self.test_images)
        # print(self.num_neg_images,self.num_pos_images)

    def __len__(self):
        if not self.testing:
            return self.num_pos_images + self.num_neg_images
        else:
            return self.num_test_images

    def __getitem__(self, idx):

        if not self.testing:

            if idx >= self.num_pos_images:
                label = 0
                image_name = self.neg_images[idx-self.num_pos_images]
            else:
                label = 1
                image_name = self.pos_images[idx]

            image = imread(image_name)
            image = self.custom_transform(image)

            return image, label, image_name

        else:
            image_name = self.test_images[idx]

            image = imread(image_name)
            image = self.custom_transform(image)

            return image, image_name





def plots(epochs, train_acc, test_acc, train_loss, test_loss,filename):
    plt.style.use('bmh')

    fig=plt.figure(figsize=(8,6))
    plt.plot(epochs,train_acc,  'r', epochs,test_acc, 'g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    fig.savefig(filename + '_accuracy.png')

    fig=plt.figure(figsize=(8,6))
    plt.plot(epochs,train_loss,  'r', epochs,test_loss, 'g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    fig.savefig(filename + '_loss.png')


    plt.close('all')



def write_csv(filename, train_acc,test_acc,train_loss,test_loss,epoch):
    if epoch==0:

        with open(filename, 'w') as f:
            f.write('train_acc,val_acc,train_loss, val_loss\n')
            f.write('{0},{1},{2},{3}\n'.format(train_acc[-1],\
                                         test_acc[-1],\
                                          train_loss[-1],\
                                           test_loss[-1],\
                                           ))

    else:
        with open(filename, 'a') as f:
            f.write('{0},{1},{2},{3}\n'.format(train_acc[-1],\
                                         test_acc[-1],\
                                          train_loss[-1],\
                                           test_loss[-1],\
                                          ))
