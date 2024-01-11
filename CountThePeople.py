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


# MY_DATA = Path(os.getenv('DATA_DIR', '../../data'), 'category4')

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

    return data.transforms.presets.yolo.transform_test(array, short=416)
    raise NotImplementedError()

norm_image, unnorm_image = transform_image(test_output)




plt.imshow(unnorm_image)
fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.show()




def detectNetwork(network, data):


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

    detected_objects = 0
    while (scores[0][detected_objects] > threshold):
        detected_objects += 1
    count = 0
    reqd_idx = network.classes.index(object_label)
    for idx in class_ids[0][:detected_objects]:
        if (idx == reqd_idx):
            count += 1
    return count





for object_label in ["person", "sports ball"]:
    count = count_object(network, class_ids, scores, bounding_boxes, object_label)
    print("{} objects of class '{}' detected".format(count, object_label))




num_people = count_object(network, class_ids, scores, bounding_boxes, "person")
assert num_people == 6




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

