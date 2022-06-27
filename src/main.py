'''
purpose_: Main Program
author_: Sanjay 
status_: development
version_: 1.0
'''

import os
from process_image import process_data, process_model, plot_output

def main(model_type: str, 
         data_path: str, 
         train_model: bool):
    """Main function starting the program

    Args:
        model_type (str): Two models - Custom model (custom_model) and pretrained model (vgg19)
        data_path (str): Path to input images under data folder
        train_model (bool): To train the model or use the pretrained model stored in models folder
    """
    
    # Process images and get training and testing data
    image_train, image_test, shape_image = process_data(data_path)

    # Predit the images
    model, embedding_data_, embedding_data_flatten = process_model(model_type, 
                                                                    shape_image, 
                                                                    image_train, 
                                                                    image_test,
                                                                    train_model)
    # Plot the output
    plot_output(model_type,
                image_train,
                model, 
                embedding_data_,
                embedding_data_flatten)

if __name__ == "__main__":
    model_type = "custom_model" 
    # print(os.getcwd())
    data_path = "data/02958343/" # Cars dataset
    train_model = False
        
    main(model_type, 
         data_path, 
         train_model)
    
