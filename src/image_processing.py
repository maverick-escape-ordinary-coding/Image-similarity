'''
purpose_: Image Processing
author_: Sanjay
status_: development
version_: 1.2
'''

from skimage.transform import resize

class ImageProcessing(object):
    
    def __init__(self, image_shape_resize):
        self.image_shape_resize = image_shape_resize
        
    def __call__(self, image_):
        image_transformed = resize_image(image_, self.image_shape_resize)
        image_transformed = normalise_image(image_transformed)
        
        return image_transformed
    
def normalise_image(image_):
    """Normalise Image Pixels

    Args:
        image_ (scikit array): Input image

    Returns:
        _type_: scikit array
        Normalised image
    """
    return image_ / 255.

def resize_image(image_, image_shape_resize):
    """Resize Image

    Args:
        image_ (scikit array): Input image
        image_shape_resize (tuple): Width and height of image

    Returns:
        _type_: scikit array
        resized image
    """
    image_resized = resize(image_, image_shape_resize,
                           anti_aliasing = True, preserve_range = True)
    assert image_resized.shape == image_shape_resize
    return image_resized

def perform_operation(images_, operation):
    """Routine to perform image processing operations on set of images

    Args:
        images_ (list): list of images
        operation (func): function to call on each image

    Returns:
        _type_: _description_
    """
    return list(map(operation, images_))
