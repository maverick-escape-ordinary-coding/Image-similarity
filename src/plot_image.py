'''
purpose_: Plot images 
author_: Sanjay 
status_: development
version_: 1.4
'''

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class Plot(object):
    
    def __init__(self, image_target, image_source):
        self.image_target = image_target
        self.image_source = image_source
        
    def __call__(self, *args):
        return plot_query_retrieval(self.image_target, self.image_source)
    
def plot_query_retrieval(self):
    """Plots images in 2 rows: top row is input image, bottom row is predicted images
    """
    num_image_source = len(self.image_source)
    fig = plt.figure(figsize=(2*num_image_source, 4))
    fig.suptitle("Image Retrieval (k={})".format(num_image_source), fontsize = 25)

    # Plot query image
    ax = plt.subplot(2, num_image_source, 0 + 1)
    plt.imshow(self.image_target)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(4)  # increase border thickness
        ax.spines[axis].set_color('black')  # set to black
    ax.set_title("query",  fontsize=14)  # set subplot title

    # Plot retrieval images
    for index, img in enumerate(self.image_source):
        ax = plt.subplot(2, num_image_source, num_image_source + index + 1)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)  # set border thickness
            ax.spines[axis].set_color('black')  # set to black
        ax.set_title("Rank #%d" % (index+1), fontsize=14)  # set subplot title

        plt.savefig('../output/', bbox_inches='tight')
    plt.close()