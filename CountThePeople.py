import mxnet as mxn
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv as gvn
import hashlib
from pylab import rcParams
from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
import numpy as np
import os
from pathlib import Path

rcParams['figure.figsize'] = 5, 10


MY_DATA = Path(os.getenv('DATA_DIR', '../../data'), 'category4')

MY_DATA = Path(os.getenv('DATA_DIR', '.'), 'category4')
MY_IMAGES = Path(MY_DATA, 'images')
MY_MODELS = Path(MY_DATA, 'models')

for my_model in gvn.model_zoo.get_model_list():
    print(my_model)

my_model_options = ['senet_154',
                 'mobilenetv3_large',
                 'faster_rcnn_fpn_resnet101_v1d_coco',
                 'yolo3_darknet53_coco',
                 'fcn_resnet101_coco',
                 'deeplab_resnet101_coco']

selected_model_option = 'yolo3_darknet53_coco'


network = gvn.model_zoo.get_model(selected_model_option, pretrained=True, root=MY_MODELS)

def stack_image(f_dir):

    return mxn.image.imread(f_dir)
    raise NotImplementedError()


test_f_dir = Path(MY_IMAGES, '32742378405_3ecc8cc958_b.jpg')

# test_f_dir = Path(MY_IMAGES, 'fun_company.jpg')
test_output = stack_image(test_f_dir)




plt.imshow(test_output.asnumpy())
fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.show()

def transform_image(array):
    """
    Should transform image by:

    1) Resizing the shortest dimension to 416. e.g (832, 3328) -> (416, 1664).
    2) Cropping to a center square of dimension (416, 416).
    3) Converting the image from HWC layout to CHW layout.
    4) Normalizing the image using COCO statistics (i.e. per colour channel mean and variance).
    5) Creating a batch of 1 image.

    :param f_dir: array (in HWC layout).
    :type f_dir: mx.nd.NDArray

    :return: a batch of a single transformed images (in NCHW layout) and a un-normalized image.
    :rtype: tuple of (mx.nd.NDArray, numpy.ndarray)
    """
    # YOUR CODE HERE
    return data.transforms.presets.yolo.transform_test(array, short=416)
    raise NotImplementedError()

norm_image, unnorm_image = transform_image(test_output)




plt.imshow(unnorm_image)
fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.show()


# ### 4) Using a my_model
#
# Your next task is to pass a transformed image through the network to obtain bounding box and class predictions from the network.
#
# We'll ignore the requirement of creating just a people detector for now.
#
# **Hint**: Don't forget that you're typically working with a batch of images, even when you only have one image.

# In[14]:


def detectNetwork(network, data):
    """
    Should return the bounding boxes and class predictions from a given network and image.

    :param network: pre-trained object detection my_model
    :type network: mx.gluon.Block
    :param data: batch of transformed images of shape (1, 3, 416, 416)
    :type data: mx.nd.NDArray

    :return: tuple of class_ids, scores, bounding_boxes
    :rtype: tuple of mxn.nd.NDArrays
    """
    # YOUR CODE HERE
    return network(data)
    raise NotImplementedError()
    return class_ids, scores, bounding_boxes


# In[15]:


class_ids, scores, bounding_boxes = detectNetwork(network, norm_image)



ax = utils.viz.plot_bbox(unnorm_image, bounding_boxes[0], scores[0], class_ids[0], class_names=network.classes)
fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.show()


def count_object(network, class_ids, scores, bounding_boxes, object_label, threshold=0.75):
    """
    Counts objects of a given type that are predicted by the network.

    :param network: object detection network
    :type network: mx.gluon.nn.Block
    :param class_ids: predicted object class indexes (e.g. 123)
    :type class_ids: mxn.nd.NDArrays
    :param scores: predicted object confidence
    :type scores: mxn.nd.NDArrays
    :param bounding_boxes: predicted object locations
    :type bounding_boxes: mxn.nd.NDArrays
    :param object_label: object to be counted (e.g. "person")
    :type object_label: str
    :param threshold: required confidence for object to be counted
    :type threshold: float

    :return: number of objects that are predicted by the network.
    :rtype: int
    """
    detected_objects = 0
    while (scores[0][detected_objects] > threshold):
        detected_objects += 1
    count = 0
    reqd_idx = network.classes.index(object_label)
    for idx in class_ids[0][:detected_objects]:
        if (idx == reqd_idx):
            count += 1
    return count


# In[18]:


for object_label in ["person", "sports ball"]:
    count = count_object(network, class_ids, scores, bounding_boxes, object_label)
    print("{} objects of class '{}' detected".format(count, object_label))

# In[19]:


num_people = count_object(network, class_ids, scores, bounding_boxes, "person")
assert num_people == 6

# In[20]:


thresholds = [0, 0.5, 0.75, 0.9, 0.99, 0.999]
for threshold in thresholds:
    num_people = count_object(network, class_ids, scores, bounding_boxes, "person", threshold=threshold)
    print("{} people detected using a threshold of {}.".format(num_people, threshold))




class PersonCounter():
    def __init__(self, threshold):
        self._network = gvn.model_zoo.get_model(selected_model_option, pretrained=True, root=MY_MODELS)
        self._threshold = threshold

    def set_threshold(self, threshold):
        self._threshold = threshold

    def count(self, filepath, visualize=False):

        norm_image, unnorm_image = transform_image(stack_image(filepath))
        class_ids, scores, bounding_boxes = network(norm_image)
        if visualize:
            self._visualize(unnorm_image, class_ids, scores, bounding_boxes)

        num_people = count_object(self._network, class_ids, scores, bounding_boxes, 'person', threshold=self._threshold)
        if num_people == 1:
            print('{} person detected in {}.'.format(num_people, filepath))
        else:
            print('{} people detected in {}.'.format(num_people, filepath))
        return num_people

    def _visualize(self, unnorm_image, class_ids, scores, bounding_boxes):
        """
        Since the transformed_image is in NCHW layout and the values are normalized,
        this method slices and transposes to give CHW as required by matplotlib,
        and scales (-2, +2) to (0, 255) linearly.
        """
        ax = utils.viz.plot_bbox(unnorm_image,
                                 bounding_boxes[0],
                                 scores[0],
                                 class_ids[0],
                                 class_names=self._network.classes)
        fig = plt.gcf()
        fig.set_size_inches(14, 14)
        plt.show()




counter = PersonCounter(threshold=0.9)

counter.set_threshold(0.5)



counter.count(Path(MY_IMAGES, '18611133536_534285f26d_b.jpg'), visualize=True)
# old aniix: counter.count(Path(MY_IMAGES, '18611133536_534285f26d_b.jpg'), visualize=True)
counter.count(Path(MY_IMAGES, 'fun_company.jpg'), visualize=True)


total_count = 0
for filepath in MY_IMAGES.glob('**/*.jpg'):
    total_count += counter.count(filepath)
print("### Summary: {} people detected.".format(total_count))

