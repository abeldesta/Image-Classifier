import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(1337)  # for reproducibility

# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
import skimage.io
from skimage.transform import resize
import os 
import PIL

train_loc = '/Users/abeldesta531/dsi/repos/Capstone-2/data/Train/'
artist = ['Vincent_Van_Gogh', 'Edgar_Degas', 'Pablo_Picasso']
train_dict = {}
for i in artist:
    os.chdir(train_loc + i)
    van_gogh = os.listdir()
    train_dict[i] = van_gogh

test_loc = '/Users/abeldesta531/dsi/repos/Capstone-2/data/Test/'
test_dict = {}
for i in artist:
    os.chdir(test_loc + i)
    van_gogh = os.listdir()
    test_dict[i] = van_gogh

os.chdir('/Users/abeldesta531/dsi/repos/Capstone-2/')


class ResizeImage(object):
    def __init__(self, files, shape):
        self.files = files
        self.shape = shape

    def folder(self, path):
        self.resized_img = []
        for i in self.files:
            new_path = path + i
            img = skimage.io.imread(new_path)
            self.resized_img.append(self.resize_image(img, self.shape))

    def resize_image(self, img, reshape, mode = 'constant'):
        return resize(image = img, output_shape = reshape, mode = mode, anti_aliasing=False)

    def save_folder(self, local, name):
        home = '/Users/abeldesta531/dsi/repos/Capstone-2/data/{0}'.format(local)
        os.chdir(home)
        os.mkdir('{0}/{1}'.format(home, 'resize_' + name))
        os.chdir('{0}/{1}'.format(home, 'resize_' + name))
        for img_name, img in zip(self.files, self.resized_img):
            plt.imsave('re_' + img_name, img)
        os.chdir('/Users/abeldesta531/dsi/repos/Capstone-2/')

def resize_image(img, reshape, mode = 'constant'):
    return resize(image = img, output_shape = reshape, mode = mode, anti_aliasing=False)

shape = (100, 100, 3)

if __name__ == "__main__":
    path = 'data/Train/Vincent_Van_Gogh/'
    # img = skimage.io.imread(path)
    # shape = (250,250,3)
    # resized = resize_image(img, shape)
    # fig, ax = plt.subplots(1,1)
    # ax.imshow(img)
    # plt.savefig('img/~vangogh.png')
    # ax.imshow(resized)
    # plt.savefig('img/reshape.png')
    # resizing = ResizeImage(train_dict['Vincent_Van_Gogh'], (100,100,3))
    # os.mkdir('/Users/abeldesta531/dsi/repos/Capstone-2/data/Train/resize_Vincent_Van_Gogh')
    # os.chdir('/Users/abeldesta531/dsi/repos/Capstone-2/data/Train/resize_Vincent_Van_Gogh')
    # plt.imsave('test.jpg', resized)
    # os.chdir('/Users/abeldesta531/dsi/repos/Capstone-2/')
    # resizing.folder(path)
    # resizing.save_folder('Train', 'Vincent_Van_Gogh')

    ##Save resized training and test photos into each folder
    for i in artist:
        #Test Images
        test_path = 'data/Test/{0}/'.format(i)
        test_resize =  ResizeImage(test_dict[i], (100,100,3))
        test_resize.folder(test_path)
        test_resize.save_folder('Test', i)
        #Training Images
        train_path = 'data/Train/{0}/'.format(i)
        train_resize = ResizeImage(train_dict[i], (100,100,3))
        train_resize.folder(train_path)
        train_resize.save_folder('Train', i)
       