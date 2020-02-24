from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.applications import Xception


import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import os
plt.style.use('ggplot')



def define_model(nb_filters, kernel_size, input_shape, pool_size):
    model = Sequential() 

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', 
                        input_shape=input_shape, name = 'conv_layer1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name = 'pool_layer1'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid', 
                        input_shape=input_shape, name = 'conv_layer2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name = 'pool_layer2'))

    model.add(Conv2D(nb_filters*2, (kernel_size[0], kernel_size[1]),
                        padding='valid', 
                        input_shape=input_shape, name = 'conv_layer4'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size, name = 'pool_layer4'))

    model.add(Flatten())
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(128)) 
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(learning_rate = .0001)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy', Precision(), Recall()])
    return model

# def create_transfer_model(input_size, n_categories, weights = ''):
#     base_model = Xception(weights=weights,
#                         include_top=False,
#                         input_shape=input_size)
    
#     model = base_model.output
#     model = GlobalAveragePooling2D()(model)
#     predictions = Dense(n_categories, activation='softmax')(model)
#     model = Model(inputs=base_model.input, outputs=predictions)
    
#     return model

# def change_trainable_layers(model, trainable_index):
#     for layer in model.layers[:trainable_index]:
#         layer.trainable = False
#     for layer in model.layers[trainable_index:]:
#         layer.trainable = True

# def print_model_properties(model, indices = 0):
#     for i, layer in enumerate(model.layers[indices:]):
#         print(f"Layer {i+indices} | Name: {layer.name} | Trainable: {layer.trainable}")



if __name__ == "__main__":
    nb_classes = 3 
    nb_epoch = 30
    img_rows, img_cols = 100, 100
    input_shape = (img_rows, img_cols, 3)
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)

    
    train_loc = os.path.abspath('data/Train/')
    test_loc = os.path.abspath('data/Test/')
    holdout_loc = os.path.abspath('data/Holdout/')

    filepath = os.path.abspath('src/best_model.pb')

    checkpoint = ModelCheckpoint(filepath, 
                    monitor='val_loss', 
                    verbose=0, 
                    save_best_only=True, 
                    save_weights_only=False, 
                    mode='auto', period=1)
    
    tbCallBack = TensorBoard(log_dir='./Graph', 
                            histogram_freq=0, 
                            write_graph=True, 
                            write_images=True)

    train_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(train_loc,
                batch_size= 5,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(100,100),
                shuffle=True)
    
    validation_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(
                test_loc,
                batch_size= 5,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(100,100),
                shuffle=True)

    holdout_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(
                holdout_loc,
                batch_size= 5,
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



    # cm = confusion_matrix(holdout_datagen.classes, y_pred)
    # sns.set(font_scale=2.5)
    # fig, ax = plt.subplots(figsize=(15,15))
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax, fmt='g');

    # # labels, title and ticks
    # ax.set_xlabel('Predicted labels');
    # ax.set_ylabel('True labels'); 
    # ax.set_title('Confusion Matrix'); 
    # ax.xaxis.set_ticklabels(class_labels); 
    # ax.yaxis.set_ticklabels(class_labels);
    

    # scp -i /.ssh/capstone_2.pem ~/Capstone-2/data/artists.csv ubuntu@ec2-34-227-223-16.compute-1.amazonaws.com