import numpy as np
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

home = '/Users/abeldesta531/dsi/repos/Capstone-2/data/Train'
artist = ['Vincent_Van_Gogh', 'Edgar_Degas', 'Pablo_Picasso']
for i in artist:

    os.chdir(i)
    van_gogh = os.listdir()





def resize_image(img_list, shape, mode = 'constant'):

    return resize(img, shape, mode)


shape = (100, 100, 3)

if __name__ == "__main__":
    path = 'data/Train/Vincent_Van_Gogh/Vincent_van_Gogh_1.jpg'
    img = skimage.io.imread(path)