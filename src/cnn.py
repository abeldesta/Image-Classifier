from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt 
import numpy as np
import os
plt.style.use('ggplot')


# def define_model(nb_filters, kernel_size, input_shape, pool_size):
    # model = Sequential() 

    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
    #                     padding='valid', 
    #                     input_shape=input_shape)) 
    # model.add(Activation('relu')) 

    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) 
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=pool_size)) 
    # model.add(Dropout(0.5)) 

    # model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='valid')) 
    # model.add(Activation('relu'))

    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.3))

    # model.add(Flatten()) 
    # print('Model flattened out to ', model.output_shape)

    
    # model.add(Dense(32)) 
    # model.add(Activation('relu'))

    # model.add(Dropout(0.3)) 

    # model.add(Dense(3)) 
    # model.add(Activation('softmax'))
    
    
    # model.compile(loss='categorical_crossentropy',
    #             optimizer='adam',
    #             metrics=['accuracy', Precision(), Recall()])
    # return model

def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential() 

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', 
                        input_shape=input_shape, name = 'conv_layer1'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size, name = 'pool_layer1'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), 
                        padding='valid', 
                        name = 'conv_layer2'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size, name = 'pool_layer2'))
    model.add(Dropout(0.5))
    model.add(Conv2D(nb_filters*2, (kernel_size[0], kernel_size[1]), 
                        padding='valid', 
                        name = 'conv_layer3'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=pool_size, name = 'pool_layer3'))
    model.add(Dropout(0.8))

    model.add(Flatten())
    print('Model flattened out to ', model.output_shape)

    
    model.add(Dense(64)) 
    model.add(Activation('relu'))

    model.add(Dropout(0.6))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy', Precision(), Recall()])
    return model



if __name__ == "__main__":
    nb_classes = 3 
    nb_epoch = 50   
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    
    train_loc = os.path.abspath('data/Train/')
    test_loc = os.path.abspath('data/Test/')
    holdout_loc = os.path.abspath('data/Holdout/')

    train_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(train_loc,
                batch_size= 50,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(100,100),
                shuffle=True)
    
    validation_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(
                test_loc,
                batch_size= 50,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(100,100),
                shuffle=True)

    holdout_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(
                holdout_loc,
                batch_size= 50,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(100,100),
                shuffle=True)



    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    hist = model.fit_generator(train_datagen,
                        steps_per_epoch=None,
                        epochs=nb_epoch, verbose=1,  
                        validation_data=validation_datagen,
                        validation_steps=None,
                        validation_freq=1,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=True,
                        shuffle=True, initial_epoch=0)

    score = model.evaluate(holdout_datagen, verbose=0)

    y_pred = model.predict_generator(holdout_datagen,
                                        workers = 1,
                                        use_multiprocessing = True,
                                        verbose = 1)

    summary = model.summary

    ##PLOTTING RESULTS

    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = np.arange(1, nb_epoch+1)

    fig, ax = plt.subplots(1,1)
    ax.plot(epochs, acc, label = 'Training Acc')
    ax.plot(epochs, val_acc, label = 'Test Acc')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy')
    plt.legend()
    plt.savefig('img/CNN_acc.png')

    fig, ax = plt.subplots(1,1)
    ax.plot(epochs, loss, label = 'Training Loss')
    ax.plot(epochs, val_loss, label = 'Test Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Model Loss')
    plt.legend()
    plt.savefig('img/CNN_loss.png')

    print('Model Score: {0}'.format(score[0]))
    print('Holdout Accuracy Score: {0}'.format(score[1]))
    print('Holdout Precision Score: {0}'.format(score[2]))
    print('Holdout Recall Score: {0}'.format(score[2]))




    

