import random
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
"""
arg_to_dict = {
        "crop":random_crop,
        "resize":image_resize,
        "elastic":elastic_transform
        }
"""
def random_crop(img, target=(96,96)):
    shape = img.shape

    dx = random.randint(0, shape[0] - target[0])
    dy = random.randint(0, shape[1] - target[1])

    if len(shape) > 2:
        return img[dx:dx + target[0], dy:dy + target[1], :]

    return img[dx:dx + target[0], dy:dy + target[1]]

def image_resize(imgs, zoom=1):
    pass

def image_rotate(imgs, angle=0):
    pass


def elastic_transform(images, param_list=None, random_state=None):    
    if param_list is None:
        param_list = [(1, 1), (5, 2), (1, 0.5), (1, 3)]

    alpha, sigma = random.choice(param_list)
    assert len(images[0].shape)==2
    shape = images[0].shape
    if random_state is None:
       random_state = np.random.RandomState(None)    

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    #print(np.mean(dx), np.std(dx), np.min(dx), np.max(dx))

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    transformed = []
    for image in images:
        new = np.zeros(shape)
        if len(shape) == 3:
            for i in range(image.shape[2]):
                new[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode="reflect").reshape(shape)
        else:
            new[:, :] = map_coordinates(image[:, :], indices, order=1, mode="reflect").reshape(shape)
        transformed.append(new)
    return transformed, alpha, sigma

def get_preprocess(preprocess_list):
    """
    preprocess_list = preprocess_list.split(",")
    transforms = []
    for preprocess in preprocess_list:
        transforms.append(arg_to_dict[preprocess])
    """
    return []

