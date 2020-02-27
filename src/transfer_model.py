from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
import os
from glob import glob

class TransferModel:
    def __init__(self, modelname, input_size, n_categories, workers = -1):
        self.savename = modelname
        self.input_shape = input_size
        self.n_categories = n_categories
        self.model = self.create_transfer_model(self.input_shape, 3)
        self.workers = workers

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

        self.train_datagen = ImageDataGenerator().flow_from_directory(self.train_folder,
                batch_size= 5,
                class_mode='categorical',
                color_mode='rgb',
                target_size=(self.input_shape[0], self.input_shape[1]),
                shuffle=False)
    
        self.validation_datagen = ImageDataGenerator().flow_from_directory(
                    self.validation_folder,
                    batch_size= 5,
                    class_mode='categorical',
                    color_mode='rgb',
                    target_size=(self.input_shape[0], self.input_shape[1]),
                    shuffle=False)

        self.holdout_datagen = ImageDataGenerator().flow_from_directory(
                    self.holdout_folder,
                    batch_size= 5,
                    class_mode='categorical',
                    color_mode='rgb',
                    target_size=(self.input_shape[0], self.input_shape[1]),
                    shuffle=False)

    def add_model_head(self, base_model, n_categories):
        """
        Takes a base model and adds a pooling and a softmax output based on the number of categories

        Args:
            base_model (keras Sequential model): model to attach head to
            n_categories (int): number of classification categories

        Returns:
            keras Sequential model: model with new head
            """

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        model = Model(inputs=base_model.input, outputs=x)
        return model

    def create_transfer_model(self, input_size, n_categories, weights = 'imagenet', model=Xception):
        """
        Creates model without top and attaches new head to it
        Args:
            input_size (tuple(int, int, int)): 3-dimensional size of input to model
            n_categories (int): number of classification categories
            weights (str or arg): weights to use for model
            model (keras Sequential model): model to use for transfer
        Returns:
            keras Sequential model: model with new head
            """
        base_model = model(weights=weights,
                        include_top=False,
                        input_shape=input_size)
        self.model = self.add_model_head(base_model, n_categories)
        opt = Adam(learning_rate = .01)
        self.model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy', Precision(), Recall()])
        self._change_trainable_layers()
        return self.model

    def _change_trainable_layers(self):
        """
        unfreezes model layers after passed index, freezes all before

        Args:
        model (keras Sequential model): model to change layers
        trainable_index(int): layer to split frozen /  unfrozen at

        Returns:
            None
            """

        for layer in self.model.layers[:131]:
            layer.trainable = False
        for layer in self.model.layers[131:]:
            layer.trainable = True

    def print_model_layers(self, indices=0):
        """
        prints model layers and whether or not they are trainable

        Args:
            model (keras classifier model): model to describe
            indices(int): layer indices to print from
        Returns:
            None
            """

        for i, layer in enumerate(self.model.layers[indices:]):
            print("Layer {} | Name: {} | Trainable: {}".format(i+indices, layer.name, layer.trainable))


    def fit(self, train_loc, validation_loc, holdout_loc):
        self._init_data(train_loc, validation_loc, holdout_loc)
        print(self.class_names)
        self._create_generators()

        self.train_labels = self.train_datagen.classes
        self.test_labels  = self.validation_datagen.classes
        self.holdout_labels = self.holdout_datagen.classes

        self.train_features = self.model.predict_generator(self.train_datagen,
                                        workers=self.workers, 
                                        use_multiprocessing=True, 
                                        verbose=1)

        self.test_features = self.model.predict_generator(self.validation_datagen,
                                        workers=self.workers, 
                                        use_multiprocessing=True, 
                                        verbose=1)

        self.holdout_features = self.model.predict_generator(self.holdout_datagen,
                                        workers=self.workers, 
                                        use_multiprocessing=True, 
                                        verbose=1)



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
                                           steps=5,
                                           use_multiprocessing=True,
                                           verbose=1)
        print("holdout loss: {} accuracy: {}".format(metrics[0], metrics[1]))
        return metrics

    def set_class_names(self):
        """
        Sets the class names, sorted by alphabetical order
        """
        names = [os.path.basename(x) for x in glob(self.train_folder + '/*')]
        return sorted(names)

def main():
    train = 'data/Train'
    holdout = 'data/Holdout'
    test = 'data/Test'
    transfer = TransferModel('transfer', (250,250,3), 3, 5)
    transfer.fit(train,test,holdout)
    print(len(transfer.train_features))
    
    

if __name__ == "__main__":
    main()
