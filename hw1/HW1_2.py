import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import utils

is_debug = True

def gaussian_pyramid(input_image, level):
    """
    Args:
        input_image (numpy array): input array
        level (int): level of pyramid

    Return:
        Gaussian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).

    output = [input_image]
    for lv in range(level):
        prev_image = output[-1]
        downsample_image = utils.down_sampling(prev_image)
        output.append(downsample_image)

    return output


def laplacian_pyramid(gaussian_pyramid):
    """
    Args:
        gaussian_pyramid (list of numpy array): result from the gaussian_pyramid function

    Return:
        laplacian pyramid (list of numpy array)
    """

    # Your code
    # Note that elements in the list must be arranged in descending order in image resolution (from big image to small image).
    output = []
    for lv in range(len(gaussian_pyramid)-1):
        output.append(utils.safe_subtract(gaussian_pyramid[lv], utils.up_sampling(gaussian_pyramid[lv+1])))
    output.append(gaussian_pyramid[len(gaussian_pyramid)-1])

    return output

def blend_images(image1, image2, mask, level):
    """
    Args:
        image1 (numpy array): background image
        image2 (numpy array): object image
        mask (numpy array): mask
        level (int): level of pyramid
    Return:
        blended image (numpy array)
    """
    # Your code
    l_image1 = laplacian_pyramid(gaussian_pyramid(image1, level))
    l_image2 = laplacian_pyramid(gaussian_pyramid(image2, level))
    g_r = gaussian_pyramid(mask, level)
    l_image1 = [elem.astype(np.uint32) for elem in l_image1]
    l_image2 = [elem.astype(np.uint32) for elem in l_image2]
    g_r = [elem.astype(np.uint32) / 255.0 for elem in g_r]

    l_mix = [g_r[i]*l_image2[i] + (1.0-g_r[i])*l_image1[i] for i in range(len(l_image1))]

    output = l_mix[-1]
    for i in range(len(l_mix)-2, -1, -1):
        output = utils.safe_add(l_mix[i], utils.up_sampling(output))

    output = np.where(output > 255.0, 255, output)
    output = output.astype(np.uint8)
    return output


if __name__ == '__main__':

    hand = np.asarray(Image.open(os.path.join('images', 'hand.jpeg')).convert('RGB'))
    flame = np.asarray(Image.open(os.path.join('images', 'flame.jpeg')).convert('RGB'))
    mask = np.asarray(Image.open(os.path.join('images', 'mask.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_2')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    level = 3


    plt.figure()
    plt.imshow(Image.open(os.path.join('images', 'direct_concat.jpeg')))
    plt.axis('off')
    plt.savefig(os.path.join(logdir, 'direct.jpeg'))
    plt.show()

    ret = gaussian_pyramid(hand, level)
    if ret is not None:
        plt.figure()
        for i in range(len(ret)):
            plt.subplot(1, len(ret), i + 1)
            plt.imshow(ret[i].astype(np.uint8))
            plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian_pyramid.jpeg'))
        plt.show()

        ret = laplacian_pyramid(ret)
        if ret is not None:
            plt.figure()
            for i in range(len(ret)):
                plt.subplot(1, len(ret), i + 1)
                plt.imshow(ret[i].astype(np.uint8))
                plt.axis('off')
            plt.savefig(os.path.join(logdir, 'laplacian_pyramid.jpeg'))
            plt.show()

    ret = blend_images(hand, flame, mask, level)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'blended.jpeg'))
        plt.show()
