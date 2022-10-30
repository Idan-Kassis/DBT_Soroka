# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 12:52:04 2022

@author: Dror
"""

# Setup
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#from resnet3d import Resnet3DBuilder
#from classification_models_3D.tfkeras import Classifiers
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model


# -----------------------------------------------------------------------------

class model_tomo_3D():
    
    def __init__(self, train_dataset, validation_dataset, test_dataset, learning_rate, epochs, input_size):
        self.lr = learning_rate
        self.epochs = epochs
        self.input_size = input_size
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        
    def get_model_architecture(self):
        """Build a 3D convolutional neural network model."""
        
        inputs = keras.Input((self.input_size[0], self.input_size[1], self.input_size[2], 1))
    
        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding="same")(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
    
        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)
    
        x = layers.Conv3D(filters=256, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.Conv3D(filters=512, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
    
        x = layers.Dense(units=256, activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.4)(x)
    
        outputs = layers.Dense(units=2, activation="sigmoid")(x)
    
        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn")
        return model
    
#    def get_model_resnet(self):
#        model = Resnet3DBuilder.build_resnet_152((self.input_size[0], self.input_size[1], self.input_size[2], 1), 2)
#        return model
        
        
    def build_model(self):
        # Pre trained model
        self.model = self.get_model_architecture()
       # Network, preprocess_input = Classifiers.get('resnet18')
        #self.model = Network(input_shape=(self.input_size[0], self.input_size[1], self.input_size[2], 1), 
        #                      weights=None)
        # Build model.
        
        #self.model = self.get_model_resnet()
        self.model.summary()
        #self.model.load_weights('3d_image_classification.h5')
        #print('num layers =', len(self.model.layers))
        
    def train_model(self):
        # Compile model.
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            self.lr , decay_steps=100000, decay_rate=0.96, staircase=True
        )
        
        #self.model = multi_gpu_model(self.model, gpus=4, cpu_relocation=True)
        
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=['accuracy'],
        )
        
        # Define callbacks.
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            "DBT_3D.h5", save_best_only=True
        )
        early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)
        
        # Get x and y from imageDataGenerator
        #for X, Y in self.train_dataset:
        #    break
        
        # Train the model, doing validation at the end of each epoch
        self.history = self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            shuffle=True,
            validation_data = self.validation_dataset,
            callbacks=[checkpoint_cb, early_stopping_cb],
        )
        #callbacks=[checkpoint_cb, early_stopping_cb]
        self.vizualizing_model_performances()
        
    def vizualizing_model_performances(self):
        fig, ax = plt.subplots(1, 2, figsize=(20, 3))
        ax = ax.ravel()
        
        for i, metric in enumerate(["accuracy", "loss"]):
            ax[i].plot(self.model.history.history[metric])
            ax[i].plot(self.model.history.history["val_" + metric])
            ax[i].set_title("Model {}".format(metric))
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel(metric)
            ax[i].legend(["train", "val"])

        plt.savefig('/mnt/md0/idank' + r'model_training.jpg')
        plt.close()
        plt.clf()

        
    def prediction(self, test_data):
        self.y_pred = self.model.predict(test_data)
        return self.y_pred