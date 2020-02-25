from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import os
from glob import glob

class TransferModel:
    def __init__(self, modelname, train_folder):
        self.savename = modelname
        self.train_folder = train_folder

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
        predictions = Dense(n_categories, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
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
        return self.model

    def fit(self, train_gen, validation_gen, holdout_gen, epochs):
        filepath = 'models/{0}.hdf5'.format(self.savename)
        checkpoint = ModelCheckpoint(filepath, 
                    monitor='val_loss', 
                    verbose=0, 
                    save_best_only=True, 
                    save_weights_only=False, 
                    mode='auto', period=1)

        self.history = self.model.fit_generator(train_gen,
                                                steps_per_epoch=None,
                                                epochs=epochs, verbose=1,  
                                                validation_data=validation_gen,
                                                validation_steps=None,
                                                validation_freq=1,
                                                callbacks = checkpoint,
                                                class_weight=None,
                                                max_queue_size=10,
                                                workers=3,
                                                use_multiprocessing=True,
                                                shuffle=True, initial_epoch=0)

        best_model = load_model(filepath)
        print('evaluating simple model')
        accuracy = self.evaluate_model(best_model, holdout_gen)
        return filepath 

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
    transfer = TransferModel('transfer', train)
    classes = transfer.set_class_names()
    return classes

if __name__ == "__main__":
    main()
