from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential() # model is a linear stack of layers (don't change)

    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', 
                        input_shape=input_shape)) #first conv. layer  KEEP
    model.add(Activation('relu')) # Activation specification necessary for Conv2D and Dense layers

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.1)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) #2nd conv. layer KEEP
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.1))

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer  KEEP
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(32)) # (only) 32 neurons in this layer, really?   KEEP
    model.add(Activation('relu'))

    model.add(Dropout(0.1)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(3)) 
    model.add(Activation('softmax')) # softmax at end to pick between classes 0-9 KEEP
    
    
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    return model



cat = ['Picasso', 'Vincent', 'Degas']

# def create_labels(img, cat):



if __name__ == "__main__":
    nb_classes = 3 
    nb_epoch = 10    
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)
    nb_filters = 100
    pool_size = (2, 2)
    kernel_size = (4, 4)

    
    train_loc = os.path.abspath('data/Train/')
    test_loc = os.path.abspath('data/Test/')

    train_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(train_loc,
                batch_size= 50,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(100,100),
                shuffle=True)
    




    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    model.fit_generator(train_datagen,
                        steps_per_epoch=None,
                        epochs=nb_epoch, verbose=1,  
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=True,
                        shuffle=True, initial_epoch=0)

    

