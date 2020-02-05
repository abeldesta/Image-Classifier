from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential() # model is a linear stack of layers (don't change)

    # note: the convolutional layers and dense layers require an activation function
    # see https://keras.io/activations/
    # and https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', 
                        input_shape=input_shape)) #first conv. layer  KEEP
    model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.5))

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(32)) # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(nb_classes)) # 10 final nodes (one for each class)  KEEP
    model.add(Activation('softmax')) # softmax at end to pick between classes 0-9 KEEP
    
    # many optimizers available, see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you KEEP loss at 'categorical_crossentropy' for this multiclass problem,
    # and KEEP metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model

datagen = ImageDataGenerator(
                            rotation_range = 40,
                            width_shift_range = 0.2,
                            height_shift_range = 0.2,
                            rescale =1/255,
                            shear_range = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True,
                            fill_mode = 'nearest')


cat = ['Picasso', 'Vincent', 'Degas']

# def create_labels(img, cat):



if __name__ == "__main__":
    nb_classes = 10  
    nb_epoch = 10    
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    nb_filters = 12
    pool_size = (2, 2)
    kernel_size = (4, 4)

    os.mkdir('/tf/dsi/repos/Capstone-2/data/Train/preview')

    img = load_img('data/Train/resize_Vincent_Van_Gogh/re_Vincent_van_Gogh_1.jpg')
    x = img_to_array(img)  
    x = x.reshape((1,) + x.shape) 
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='data/Train/preview', save_prefix='Vincent_van_Gogh', save_format='jpg'):
        i += 1
        if i > 20:
            break 

