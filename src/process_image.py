'''
purpose_: Process images and predict 
author_: Sanjay 
status_: development
version_: 1.4
'''

import os
import numpy as np
import skimage.io as io
import sys
# sys.path.append('/Users/sanjaynaidu/work/challenges/kaedim/') 
import tensorflow as tf
from src.model_construction import Model
from src.image_processing import perform_operation, ImageProcessing
from src.plot_image import plot_query_retrieval
from sklearn.neighbors import NearestNeighbors

def get_data(path: str):
    """Extract all the PNG images from the given path

    Args:
        path (str): Path to data directory

    Returns:
        _type_: list of strings
        Paths to each images in the directory
    """
    
    list_files = []
    for directory_nam, _, files in os.walk(path):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension == '.png':
                list_files.append(io.imread(os.path.join(directory_nam, file), as_gray=False))
    
    return list_files

def process_data(path: str):
    """Process images and get training and testing data

    Args:
        path (str): Path to data directory

    Returns:
        _type_: 3 ndarrays
        Training images, testing images, and Shape of image
    """
    # File Paths
    data_path = os.path.join(path)

    # Read images
    print("Reading images from - {}".format(data_path))
    images_ = get_data(data_path)
    shape_image = images_[0].shape
    print("Image shape = {}".format(shape_image))  
    
    # Image Processing
    print("Image Processing...")
    image_processing = ImageProcessing(shape_image)
    len_split = int(len(images_) * 0.8)
    image_train = perform_operation(images_[1:len_split], image_processing) 
    image_test = perform_operation(images_[len_split+1:], image_processing) 

    return image_train, image_test, shape_image

def process_model(model_type: str, 
                  shape_image: tuple, 
                  image_train: np.ndarray, 
                  image_test: np.ndarray,
                  train_model: bool):  
    """Process model and get embeddings
    
    Args:
        model_type (str): Two models - Custom model (custom_model) and pretrained model (vgg19)
        shape_image (tuple): Shape of image
        image_train (np.ndarray): Numpy array of training images
        image_test (np.ndarray): Numpy array of testing images
        train_model (bool): To train the model or use the pretrained model stored in models folder

    Returns:
        _type_: 
        Model, Original embeddings and Flattened embeddings
    """
    # Model Construction
    model = Model(shape_image)
    model.set_model_arc()
    
    if model_type == "custom_model":
        input_shape_model = tuple([int(x) for x in model.model_encoder.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.model_encoder.output.shape[1:]])
        epochs = 500
        
    elif model_type == "vgg19":

        print("Loading VGG19 pre-trained model...")
        model = tf.keras.applications.VGG19(weights='imagenet', 
                                            include_top=False,
                                            input_shape=(137, 137, 3))
        model.summary()

        shape_img_resize = tuple([int(x) for x in model.input.shape[1:]])
        input_shape_model = tuple([int(x) for x in model.input.shape[1:]])
        output_shape_model = tuple([int(x) for x in model.output.shape[1:]])
        epochs = None 
        
    else:
        raise Exception("Invalid model type")            
    
    # Print model info
    print("input_shape_model = {}".format(input_shape_model))
    print("output_shape_model = {}".format(output_shape_model))
        
    # Convert images to numpy array
    X_train = np.array(image_train).reshape((-1,) + input_shape_model)
    X_test = np.array(image_test).reshape((-1,) + input_shape_model)
    print(" -> X_train.shape = {}".format(X_train.shape))
    print(" -> X_test.shape = {}".format(X_test.shape))

    # Train or Load model
    if model_type == "custom_model":
        if train_model:
            model.compile(loss="binary_crossentropy", optimizer="adam")
            model.fit(X_train, n_epochs=epochs, batch_size=256)
            model.save_models()
        else:
            model.load_models(loss="binary_crossentropy", optimizer="adam")

    # Create embeddings using model
    print("Inferencing embeddings using pre-trained model")
    embedding_train_ = model.predict(X_train)
    embedding_train_flatten = embedding_train_.reshape((-1, np.prod(output_shape_model)))
    embedding_test_ = model.predict(X_test)
    embedding_test_flatten = embedding_test_.reshape((-1, np.prod(output_shape_model)))
    print(" -> E_train.shape = {}".format(embedding_train_.shape))
    print(" -> E_test.shape = {}".format(embedding_test_.shape))
    print(" -> E_train_flatten.shape = {}".format(embedding_train_flatten.shape))
    print(" -> E_test_flatten.shape = {}".format(embedding_test_flatten.shape))

    return model, embedding_train_, embedding_train_flatten
    
def plot_output(model_type: str,
                images_train_: np.ndarray,
                model: tf.keras.models.Model, 
                embedding_train_: np.ndarray,
                embedding_train_flatten: np.ndarray):
    """Plot output of model
    
    Args:
        model_type (str): Two models - Custom model (custom_model) and pretrained model (vgg19)
        image_train (np.ndarray): Numpy array of training images
        model (tf.keras.models.Model): model object
        embedding_train_(np.ndarray): Numpy array of training embeddings
        embedding_train_flatten (np.ndarray): Numpy array of training embeddings flattened

    """
    
    # Make reconstruction visualizations
    if model_type == "custom_model":
        print("Printing output...")
        images_train_reconstruct = model.decoder.predict(embedding_train_)
        plot_query_retrieval(images_train_, images_train_reconstruct,
                            os.path.join('../output/', "{}_reconstruct.png".format(model_type)),
                            range_imgs=[0, 255],
                            range_imgs_reconstruct=[0, 1])

    # Fit kNN model on training images
    print("Fitting k-nearest-neighbour model on images...")
    knn = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn.fit(embedding_train_flatten)    