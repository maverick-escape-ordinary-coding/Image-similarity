'''
purpose_: Construction, Saving, Loading, and Training of the model
author_: Sanjay 
status_: development
version_: 1.4
'''

import tensorflow as tf
import numpy as np
from src.split_indices import split

class Model():

    def __init__(self, shape_img):
        """Construction, Saving, Loading, and Training of the model

        Args:
            shape_img (_type_): Shape of image required for model
        """
        self.shape_img = shape_img
        self.model = None
        self.model_encode = None
        self.model_decode = None

    def fit(self, X, epochs = 10, batch_size = 100):
        """Model Training

        Args:
            X (ndarray): Input Data
            epochs (int, optional): _description_. Number of passes
            batch_size (int, optional): _description_. Number of data batches
        """
        # Get data indices
        indices_fracs = split(fracs=[0.8, 0.2], N=len(X), seed=0)
        
        # Split the data
        X_train, X_valid = X[indices_fracs[0]], X[indices_fracs[1]]
        
        # Fit the model
        self.model.fit(X_train, X_train,
                             epochs = epochs,
                             batch_size = batch_size,
                             shuffle = True,
                             validation_data = (X_valid, X_valid))

    def predict(self, X):
        """ Model Inference

        Args:
            X (ndarray): Input Data

        Returns:
            _type_: ndarray
            Predict Image
        """
        return self.model_encode.predict(X)

    def set_model_arc(self):
        """Model Architecture
        """

        shape_img_flattened = (np.prod(list(self.shape_img)),)

        # Set encoder and decoder graphs
        hidden_layer_1, hidden_layer_2, hidden_layer_3 = 16, 8, 8
        
        # convolution kernel
        convkernel = (3, 3)  
        
        # pooling kernel
        poolkernel = (2, 2)  

        input_layer = tf.keras.layers.Input(shape=self.shape_img)
        layer_x = tf.keras.layers.Conv2D(hidden_layer_1, convkernel, activation='relu', padding='same')(input_layer)
        layer_x = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(layer_x)
        layer_x = tf.keras.layers.Conv2D(hidden_layer_2, convkernel, activation='relu', padding='same')(layer_x)
        layer_x = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(layer_x)
        layer_x = tf.keras.layers.Conv2D(hidden_layer_3, convkernel, activation='relu', padding='same')(layer_x)
        encoded = tf.keras.layers.MaxPooling2D(poolkernel, padding='same')(layer_x)

        layer_x = tf.keras.layers.Conv2D(hidden_layer_3, convkernel, activation='relu', padding='same')(encoded)
        layer_x = tf.keras.layers.UpSampling2D(poolkernel)(layer_x)
        layer_x = tf.keras.layers.Conv2D(hidden_layer_2, convkernel, activation='relu', padding='same')(layer_x)
        layer_x = tf.keras.layers.UpSampling2D(poolkernel)(layer_x)
        layer_x = tf.keras.layers.Conv2D(hidden_layer_1, convkernel, activation='relu')(layer_x)
        layer_x = tf.keras.layers.UpSampling2D(poolkernel)(layer_x)
        decoded = tf.keras.layers.Conv2D(self.shape_img[2], convkernel, activation='sigmoid', padding='same')(layer_x)

        # Autoencoder model
        model_ = tf.keras.Model(input, decoded)
        input_autoencoder_shape = model_.layers[0].input_shape[1:]
        output_autoencoder_shape = model_.layers[-1].output_shape[1:]

        # Encoder model
        encoder = tf.keras.Model(input, encoded)  # set encoder
        input_encoder_shape = encoder.layers[0].input_shape[1:]
        output_encoder_shape = encoder.layers[-1].output_shape[1:]

        # Decoder model
        decoded_input = tf.keras.Input(shape=output_encoder_shape)
        decoded_output = model_.layers[-7](decoded_input)  # Conv2D
        decoded_output = model_.layers[-6](decoded_output)  # UpSampling2D
        decoded_output = model_.layers[-5](decoded_output)  # Conv2D
        decoded_output = model_.layers[-4](decoded_output)  # UpSampling2D
        decoded_output = model_.layers[-3](decoded_output)  # Conv2D
        decoded_output = model_.layers[-2](decoded_output)  # UpSampling2D
        decoded_output = model_.layers[-1](decoded_output)  # Conv2D

        decoder = tf.keras.Model(decoded_input, decoded_output)

        # Generate summaries
        print("\nmodel summary():{}".format(model_.summary()))
        print("\nencoder summary():{}".format(encoder.summary()))
        print("\ndecoder summary():{}".format(decoder.summary()))

        # Assign models
        self.model = model_
        self.model_encode = encoder
        self.model_decode = decoder

    def compile(self, loss="binary_crossentropy", optimizer="adam"):
        """Compile Model

        Args:
            loss (str, optional): _description_. Type of penalty
            optimizer (str, optional): _description_. Type of weights update
        """
        self.model.compile(optimizer=optimizer, loss=loss)

    def load_models(self, loss="binary_crossentropy", optimizer="adam"):
        """Load model architecture and weights

        Args:
            loss (str, optional): _description_. Type of penalty
            optimizer (str, optional): _description_. Type of weights update
        """
        self.model = tf.keras.models.load_model('../models/autoencoder.h5')
        self.model_encode = tf.keras.models.load_model('../models/encoder.h5')
        self.model_decode = tf.keras.models.load_model('../models/decoder.h5')
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model_encode.compile(optimizer=optimizer, loss=loss)
        self.model_decode.compile(optimizer=optimizer, loss=loss)

    def save_models(self):
        """Save Model (architecture and weights) to folder Models
        """
        self.model.save('../models/autoencoder.h5')
        self.model_encode.save('../models/encoder.h5')
        self.model_decode.save('../models/decoder.h5')