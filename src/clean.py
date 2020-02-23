import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras 
import skimage.io
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import PIL
np.random.seed(1337)  # for reproducibility


home = os.getcwd()

train_loc = 'data/Train/'
artist = ['Edgar_Degas', 'Pablo_Picasso', 'Vincent_Van_Gogh']
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

holdout_loc = 'data/Holdout/'
hold_dict = {}
for i in artist:
    os.chdir(os.path.abspath(holdout_loc + i))
    van_gogh = os.listdir()
    hold_dict[i] = van_gogh
    os.chdir(home)

os.chdir(home)


class ImagePipeline(object):
    '''
    Pipeline with methods to resize images, create datagens, move/delete necessary 
    folders.
    '''

    def __init__(self, folder, shape, home, import_path, export_path):
        '''
        Initalize class basic attributes

        Args:
        folder (str): path to artist image folder
        shape (tuple(int,int,int)): size of image for RGB inputs
        home (str): absolute path
        import_path (str): path for images to augment
        export_path (str): path for augmented images
        '''
        self.folder = folder
        self.shape = shape
        self.num_images = len(folder)*20
        self.home = home
        self.export_path = export_path
        self.import_path = import_path
        self.data_generator()
        try:
            os.mkdir(export_path)
        except:
            pass

    def image_folder(self, path):
        '''
        Create a folder of resized images

        Args:
            path (str): path to artists image folder

        Returns:
            list: list of resized images
        '''
        self.resized_img = []
        for i in self.folder:
            new_path = path + i
            img = skimage.io.imread(new_path)
            self.resized_img.append(self.resize_image(img, self.shape))

    def resize_image(self, img, mode = 'constant'):
        '''
        Resize image
        
        Args:
            img (array): matrix of pixel intensities for image
            mode (str): parameter for the skimage resize function
        
        Returns:
            array: array of pixel intensity for resized image
        '''
        return resize(image = img, 
                        output_shape = self.shape, 
                        mode = mode, 
                        anti_aliasing=False)

    def save_folder(self, local, name):
        '''
        Saves list of resized images to folder

        Args:
            local (str): location for resized images folder
            name (str): name of artist
        '''

        os.chdir(self.home)
        os.mkdir(os.path.abspath(local + '/resize_' + name))
        os.chdir(os.path.abspath(local + '/resize_' + name))
        for img_name, img in zip(self.folder, self.resized_img):
            # img_name = img_name.replace('.jpg', '.png')
            skimage.io.imsave('re_' + img_name, img)
        os.chdir(self.home)

    def delete_move_folder(self, local, dest, rm_dir):
        '''
        Move resized and augmented images to proper folder
        and delete folder from original location

        Args:
            local (str): path to location of folder
            dest (str): path for destination of folder
            rm_dir (str): path of direcotry to delete
        '''
        os.system('rm -rf ' + rm_dir)
        os.system('mkdir ' + dest)
        os.system('mv ' + os.path.abspath(local) + ' ' + os.path.abspath(dest))
        os.mkdir(self.export_path)

    def data_generator(self):
        '''
        Create image generator for image augmentation 
        '''
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
        '''
        Augment each image to create more data
        '''
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
    # for i in artist:
    #     #Test Images
    #     test_path = 'data/Test/{0}/'.format(i)
    #     test_resize =  ImagePipeline(test_dict[i], 
    #                                     (100,100,3), 
    #                                     home, 
    #                                     import_path, 
    #                                     export_path)
    #     test_resize.image_folder(test_path)
    #     test_resize.save_folder('data/Test', i)
    #     test_resize.save_folder('data/Keras_Images', i)
    #     test_resize.image_augmentation()
    #     test_resize.delete_move_folder('data/Keras_Images' , 
    #                                     'data/Test/resize_{0}/generated_imgs'.format(i), 
    #                                     'data/Keras_Images/resize_{0}'.format(i))
    #     #Training Images
    #     train_path = 'data/Train/{0}/'.format(i)
    #     train_resize = ImagePipeline(train_dict[i], 
    #                                     (100,100,3), 
    #                                     home, 
    #                                     import_path, 
    #                                     export_path)
    #     train_resize.image_folder(train_path)
    #     train_resize.save_folder('data/Train', i)
    #     train_resize.save_folder('data/Keras_Images', i)
    #     train_resize.image_augmentation()
    #     train_resize.delete_move_folder('data/Keras_Images', 
    #                                         'data/Train/resize_{0}/generated_imgs'.format(i), 
    #                                         'data/Keras_Images/resize_{0}'.format(i))
    #     #Holdout Images
    #     holdout_path = 'data/Holdout/{0}/'.format(i)
    #     holdout_resize = ImagePipeline(hold_dict[i], 
    #                                         (100,100,3), 
    #                                         home, 
    #                                         import_path, 
    #                                         export_path)
    #     holdout_resize.image_folder(holdout_path)
    #     holdout_resize.save_folder('data/Holdout', i)
    #     holdout_resize.save_folder('data/Keras_Images', i)
    #     holdout_resize.image_augmentation()
    #     holdout_resize.delete_move_folder('data/Keras_Images', 
    #                                         'data/Holdout/resize_{0}/generated_imgs'.format(i), 
    #                                         'data/Keras_Images/resize_{0}'.format(i))

