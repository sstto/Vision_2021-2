import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def reflect_padding(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of three int [height, width]): filter size (e.g. (3,3))
    Return:
        padded image (numpy array)
    """

    # Your code

    size_x = (size[0]-1)//2
    size_y = (size[1]-1)//2

    shape = input_image[...,0].shape
    output_shape = tuple(np.add(shape, size)-1)

    output_image = np.zeros(output_shape + (3,), 'uint8')

    for rgb in range(3):
        rgb_image = input_image[..., rgb]
        #print(rgb_image)
        left = np.flip(rgb_image[..., 1:size_y+1], axis=1)
        right = np.flip(rgb_image[..., -size_y-1:-1], axis=1)
        rgb_image = np.hstack((left, rgb_image))
        rgb_image = np.hstack((rgb_image, right))
        #print(rgb_image)

        up = np.flip(rgb_image[1:size_x+1, ...], axis=0)
        down = np.flip(rgb_image[-size_x-1:-1, ...], axis=0)
        rgb_image = np.vstack((up, rgb_image))
        rgb_image = np.vstack((rgb_image, down))

        #print(rgb_image.shape)
        output_image[..., rgb] = rgb_image

    return output_image


def convolve(input_image, Kernel):
    """
    Args:
        input_image (numpy array): input array
        Kernel (numpy array): kernel shape of (height, width)
    Return:
        convolved image (numpy array)
    """
    shape = input_image.shape
    rgb_shape = input_image[..., 0].shape

    input_image = reflect_padding(input_image, Kernel.shape)
    output_image = np.zeros(shape, 'uint8')


    #flip Kernel
    Kernel = np.flip(Kernel)

    size_x, size_y = np.shape(input_image[..., 0])
    kernel_x, kernel_y = np.shape(Kernel)
    for rgb in range(3):
        rgb_image = input_image[..., rgb]
        result = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                result.append((rgb_image[i:i+kernel_x, j:j+kernel_y]*Kernel).sum())
        result = np.array(result).reshape(rgb_shape)
        output_image[..., rgb] = result


    # Your code
    # Note that the dimension of the input_image and the Kernel are different.
    # shape of input_image: (height, width, channel)
    # shape of Kernel: (height, width)
    # Make sure that the same Kernel be applied to each of the channels of the input_image
    return output_image


def median_filter(input_image, size):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,3))
    Return:
        Median filtered image (numpy array)
    """
    for s in size:
        if s % 2 == 0:
            raise Exception("size must be odd for median filter")
    shape = input_image.shape
    rgb_shape = input_image[..., 0].shape
    input_image = reflect_padding(input_image, size)
    output_image = np.zeros(shape, 'uint8')
    size_x, size_y = np.shape(input_image[...,0])
    kernel_x, kernel_y = size

    for rgb in range(3):
        rgb_image = input_image[..., rgb]
        result = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                result.append(np.median(rgb_image[i:i+kernel_x, j:j+kernel_y]))
        result = np.array(result).reshape(rgb_shape)
        output_image[..., rgb] = result

    # Your code
    return output_image


def gaussian_filter(input_image, size, sigmax, sigmay):
    """
    Args:
        input_image (numpy array): input array
        size (tuple of two int [height, width]): filter size (e.g. (3,.3))
        sigmax (float): standard deviation in X directionR
        sigmay (float): standard deviation in Y direction
    Return:
        Gaussian filtered image (numpy array)
    """
    # Your code

    shape = input_image.shape
    rgb_shape = input_image[..., 0].shape
    input_image = reflect_padding(input_image, size)
    output_image = np.zeros(shape, 'uint8')

    filter_x = np.array([(1.0/np.sqrt(2 * np.pi * (sigmax**2))) * np.exp(-((i-(size[0] // 2))**2) / (2 * (sigmax**2))) for i in range(size[0])])
    filter_x = filter_x / filter_x.sum()
    filter_y = np.array([(1.0/np.sqrt(2 * np.pi * (sigmay**2))) * np.exp(-((i-(size[1] // 2))**2) / (2 * (sigmay**2))) for i in range(size[1])])
    filter_y = filter_y / filter_y.sum()

    tmp_image = np.zeros((shape[0], input_image.shape[1], 3), dtype=filter_x.dtype)

    for rgb in range(3):
        for y in range(tmp_image.shape[1]):
            for x in range(tmp_image.shape[0]):
                tmp_image[x, y, rgb] = (input_image[x:(x+size[0]), y, rgb] * filter_x).sum()

    for rgb in range(3):
        for y in range(output_image.shape[1]):
            for x in range(output_image.shape[0]):
                output_image[x, y, rgb] = (tmp_image[x, y:(y+size[1]), rgb] * filter_y).sum()

    return output_image


if __name__ == '__main__':

    # image = np.asarray(Image.open(os.path.join('images', 'baboon.jpeg')).convert('RGB'))
    image = np.asarray(Image.open(os.path.join('images', 'gaussian_noise.jpeg')).convert('RGB'))
    # image = np.asarray(Image.open(os.path.join('images', 'salt_and_pepper_noise.jpeg')).convert('RGB'))

    logdir = os.path.join('results', 'HW1_1')
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    kernel_1 = np.ones((5,5)) / 25.
    sigmax, sigmay = 5, 5
    ret = reflect_padding(image.copy(), kernel_1.shape)

    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'reflect.jpeg'))
        plt.show()

    ret = convolve(image.copy(), kernel_1)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'convolve.jpeg'))
        plt.show()

    ret = median_filter(image.copy(), kernel_1.shape)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'median.jpeg'))
        plt.show()

    ret = gaussian_filter(image.copy(), kernel_1.shape, sigmax, sigmay)
    if ret is not None:
        plt.figure()
        plt.imshow(ret.astype(np.uint8))
        plt.axis('off')
        plt.savefig(os.path.join(logdir, 'gaussian.jpeg'))
        plt.show()


