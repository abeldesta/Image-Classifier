import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import skimage.io
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import PIL
np.random.seed(1337)  # for reproducibility

# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical

home = os.getcwd()

train_loc = 'data/Train/'
artist = ['Vincent_Van_Gogh', 'Edgar_Degas', 'Pablo_Picasso']
train_dict = {}
for i in artist:
    os.chdir(os.path.abspath(train_loc + i))
    van_gogh = os.listdir()
    train_dict[i] = van_gogh
    os.chdir(home)

test_loc = 'data/Test/'
test_dict = {}
for i in artist:
    os.chdir(os.path.abspath(test_loc + i))
    van_gogh = os.listdir()
    test_dict[i] = van_gogh
    os.chdir(home)

os.chdir(home)


class ImagePipeline(object):
    def __init__(self, files, shape, import_path, export_path):
        self.files = files
        self.shape = shape
        self.num_images = 1000
        self.export_path = export_path
        self.import_path = import_path
        self.data_generator()
        try:
            os.mkdir(export_path)
        except:
            pass

    def folder(self, path):
        self.resized_img = []
        for i in self.files:
            new_path = path + i
            img = skimage.io.imread(new_path)
            self.resized_img.append(self.resize_image(img, self.shape))

    def resize_image(self, img, reshape, mode = 'constant'):
        return resize(image = img, output_shape = reshape, mode = mode, anti_aliasing=False)

    def save_folder(self, local, name):
        home = '/tf/dsi/repos/Capstone-2/data/'
        os.chdir(home)
        os.mkdir(os.path.abspath(local + '/resize_' + name))
        os.chdir(os.path.abspath(local + '/resize_' + name))
        for img_name, img in zip(self.files, self.resized_img):
            # img_name = img_name.replace('.jpg', '.png')
            skimage.io.imsave('re_' + img_name, img)
        os.chdir('/tf/dsi/repos/Capstone-2/')

    def delete_move_folder(self, local, dest, rm_dir):
        os.system('rm -rf ' + rm_dir)
        os.system('mkdir ' + dest)
        os.system('mv ' + os.path.abspath(local) + ' ' + os.path.abspath(dest))
        os.mkdir(self.export_path)

    def data_generator(self):
        self.datagen = ImageDataGenerator(
                                rotation_range = 30,
                                width_shift_range = 0.2,
                                height_shift_range = 0.2,
                                rescale =1/255,
                                shear_range = 0.0,
                                zoom_range = 0.2,
                                horizontal_flip = True,
                                fill_mode = 'nearest')
    
    def image_augmentation(self):
        i = 0
        for batch in self.datagen.flow_from_directory(
                                        directory=self.import_path,
                                        save_to_dir=self.export_path,
                                        save_prefix='keras_',
                                        save_format='jpg',
                                        batch_size=1, color_mode='rgb'):
            i += 1
            if i == self.num_images:
                break

shape = (100, 100, 3)


if __name__ == "__main__":
    shape = (100, 100, 3)
    import_path = 'data/Keras_Images'
    export_path = 'data/Keras_Images'
    ##Save resized training and test photos into each folder
    for i in artist:
        #Test Images
        test_path = 'data/Test/{0}/'.format(i)
        test_resize =  ImagePipeline(test_dict[i], (100,100,3),import_path, export_path)
        test_resize.folder(test_path)
        test_resize.save_folder('Test', i)
        test_resize.save_folder('Keras_Images', i)
        test_resize.image_augmentation()
        test_resize.delete_move_folder('data/Keras_Images' , 'data/Test/resize_{0}/generated_imgs'.format(i), 'data/Keras_Images/resize_{0}'.format(i))
        #Training Images
        train_path = 'data/Train/{0}/'.format(i)
        train_resize = ImagePipeline(train_dict[i], (100,100,3), import_path, export_path)
        train_resize.folder(train_path)
        train_resize.save_folder('Train', i)
        train_resize.save_folder('Keras_Images', i)
        train_resize.image_augmentation()
        train_resize.delete_move_folder('data/Keras_Images', 'data/Train/resize_{0}/generated_imgs'.format(i), 'data/Keras_Images/resize_{0}'.format(i))


    # test_path = 'data/Test/Pablo_Picasso/'
    # import_path = 'data/Keras_Images'
    # export_path = 'data/Keras_Images'
    # test_resize =  ImagePipeline(test_dict['Pablo_Picasso'], (100,100,3), export_path, export_path)
    # test_resize.folder(test_path)
    # test_resize.save_folder('Test', 'Pablo_Picasso')
    # test_resize.save_folder('Keras_Images', 'Pablo_Picasso')
    # test_resize.image_augmentation()
    # test_resize.delete_move_folder('data/Keras_Images' , 'data/Test/resize_Pablo_Picasso/generated_imgs', 'data/Keras_Images/resize_Pablo_Picasso')

    