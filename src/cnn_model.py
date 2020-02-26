from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import Xception
from transfer_model import main
import matplotlib.pyplot as plt 
from glob import glob
import numpy as np
# import seaborn as sns 
import os
plt.style.use('ggplot')

class SimpleCNN:
    
    '''
    Simple Convolution Neural Network class with methods to create generators, fit, and evaluate model
    '''
    def __init__(self, nb_classes, modelname, workers = 1, nb_epoch = 10, nb_filters = 32, kernel_size = (3,3), input_shape = (100,100,3), pool_size = (2,2)):
        self.nb_classes = nb_classes
        self.nb_epoch = nb_epoch
        self.img_rows, self.img_cols = 100, 100
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.savename = modelname 
        self.workers = workers
        self.model = self.define_model()

    def _init_data(self, train_folder, validation_folder, holdout_folder):
        """
        Initializes class data

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            holdout_folder(str): folder containing holdout data
            """
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.holdout_folder = holdout_folder

        self.nTrain = sum(len(files) for _, _, files in os.walk(self.train_folder)) 
        self.nVal = sum(len(files) for _, _, files in os.walk(self.validation_folder))
        self.nHoldout = sum(len(files) for _, _, files in os.walk(self.holdout_folder))
        self.n_categories = sum(len(dirnames) for _, dirnames, _ in os.walk(self.train_folder))
        self.class_names = self.set_class_names()
    
    def _create_generators(self):
        '''
        Creates generators to create augmented images from directed
        '''

        self.train_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(self.train_folder,
                batch_size= 5,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(100,100),
                shuffle=False)
    
        self.validation_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(
                    self.validation_folder,
                    batch_size= 5,
                    class_mode='categorical',
                    color_mode='rgb',
                    target_size=(100,100),
                    shuffle=False)

        self.holdout_datagen = ImageDataGenerator(rescale =1./255).flow_from_directory(
                    self.holdout_folder,
                    batch_size= 5,
                    class_mode='categorical',
                    color_mode='rgb',
                    target_size=(100,100),
                    shuffle=False)

    def define_model(self):
        '''
        Define CNN model
        '''

        model = Sequential() 

        model.add(Conv2D(self.nb_filters, (self.kernel_size[0], self.kernel_size[1]),
                            padding='valid', 
                            input_shape=self.input_shape, name = 'conv_layer1'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size, name = 'pool_layer1'))

        model.add(Conv2D(self.nb_filters, (self.kernel_size[0], self.kernel_size[1]),
                            padding='valid', 
                            input_shape=self.input_shape, name = 'conv_layer2'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size, name = 'pool_layer2'))

        model.add(Conv2D(self.nb_filters*2, (kernel_size[0], kernel_size[1]),
                            padding='valid', 
                            input_shape=self.input_shape, name = 'conv_layer4'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=self.pool_size, name = 'pool_layer4'))

        model.add(Flatten())
        print('Model flattened out to ', model.output_shape)

        model.add(Dense(128)) 
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Dropout(0.5))
        model.add(Activation('softmax'))
        opt = Adam(lr = .001)
        model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy', Precision(), Recall()])
        return model

    def fit(self, train_folder, validation_folder, holdout_folder):
        '''
        Fits CNN to the data, then saves and predicts on best model

        Args:
            train_folder(str): folder containing train data
            validation_folder(str): folder containing validation data
            holdout_folder(str): folder containing holdout data            

        Returns:
            str: file path for best model
        '''
        self._init_data(train_folder, validation_folder, holdout_folder)
        print(self.class_names)
        self._create_generators()

        filepath = 'models/{0}.hdf5'.format(self.savename)
        checkpoint = ModelCheckpoint(filepath, 
                    monitor='val_loss', 
                    verbose=0, 
                    save_best_only=True, 
                    save_weights_only=False, 
                    mode='auto', period=1)

        hist = self.model.fit_generator(self.train_datagen,
                        steps_per_epoch=None,
                        epochs=self.nb_epoch, verbose=1,  
                        validation_data=self.validation_datagen,
                        validation_steps=None,
                        validation_freq=1,
                        callbacks = [checkpoint],
                        class_weight=None,
                        max_queue_size=10,
                        workers=self.workers,
                        use_multiprocessing=True,
                        shuffle=True, initial_epoch=0)

        best_model = load_model(filepath)
        print('evaluating simple model')
        accuracy = self.evaluate_model(best_model, self.holdout_datagen)
        return self.savename
    
    def evaluate_model(self, model, holdout_gen):
        """
        evaluates model on holdout data
        Args:
            model (keras classifier model): model to evaluate
            holdout_folder (str): path of holdout data
        Returns:
            list(float): metrics returned by the model, typically [loss, accuracy]
            """

        metrics = model.evaluate_generator(holdout_gen,
                                           use_multiprocessing=True,
                                           verbose=1)
        print('Model Score: {0}'.format(metrics[0]))
        print('Holdout Accuracy Score: {0}'.format(metrics[1]))
        print('Holdout Precision Score: {0}'.format(metrics[2]))
        print('Holdout Recall Score: {0}'.format(metrics[3]))
        return metrics

    def set_class_names(self):
        """
        Sets the class names, sorted by alphabetical order
        """
        names = [os.path.basename(x) for x in glob(self.train_folder + '/*')]
        return sorted(names)





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

    cnn = SimpleCNN(nb_classes = nb_classes, modelname = 'simpleCNN')
    cnn.fit(train_loc, test_loc, holdout_loc)


    
    
    # y_pred = model.predict_generator(holdout_datagen,
    #                                     workers = 1,
    #                                     use_multiprocessing = True,
    #                                     verbose = 1)

    # summary = model.summary

    # ##PLOTTING RESULTS

    # acc = hist.history['acc']
    # val_acc = hist.history['val_acc']
    # loss = hist.history['loss']
    # val_loss = hist.history['val_loss']
    # epochs = np.arange(1, nb_epoch+1)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(epochs, acc, label = 'Training Acc')
    # ax.plot(epochs, val_acc, label = 'Test Acc')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Accuracy')
    # ax.set_title('Model Accuracy')
    # plt.legend()
    # plt.savefig('img/CNN_acc.png')

    # fig, ax = plt.subplots(1,1)
    # ax.plot(epochs, loss, label = 'Training Loss')
    # ax.plot(epochs, val_loss, label = 'Test Loss')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Loss')
    # ax.set_title('Model Loss')
    # plt.legend()
    # plt.savefig('img/CNN_loss.png')

    # print('Model Score: {0}'.format(score[0]))
    # print('Holdout Accuracy Score: {0}'.format(score[1]))
    # print('Holdout Precision Score: {0}'.format(score[2]))
    # print('Holdout Recall Score: {0}'.format(score[2]))



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