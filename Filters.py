import numpy as np
import math
import copy


def filt_gauss(sigma=0.1):
    matrix = np.array([
        [math.exp(- 2 / math.pow(sigma, 2)), math.exp(- 1 / (2 * math.pow(sigma, 2))),
         math.exp(- 2 / math.pow(sigma, 2))],
        [math.exp(- 1 / (2 * math.pow(sigma, 2))), 1, math.exp(- 1 / (2 * math.pow(sigma, 2)))],
        [math.exp(- 2 / math.pow(sigma, 2)), math.exp(- 1 / (2 * math.pow(sigma, 2))),
         math.exp(- 2 / math.pow(sigma, 2))],

    ])
    return (1 / (2 * (math.pow(sigma, 2)) * math.pi)) * matrix


def filter_gaussien(img, sigma=1):
    kernel = filt_gauss(sigma)
    image = copy.copy(img)
    new_image = np.zeros([image.shape[0] + 2, image.shape[1] + 2])

    new_image[1:-1, 1:-1] = image

    output = np.zeros([image.shape[0], image.shape[1]])

    for i in range(1, new_image.shape[0] - 1):
        for j in range(1, new_image.shape[1] - 1):
            output[i - 1][j - 1] = (kernel[0][0] * new_image[i - 1][j - 1]) + (kernel[0][1] * new_image[i - 1][j]) + (
                    kernel[0][2] * new_image[i - 1][j + 1]) + (kernel[1][0] * new_image[i][j - 1]) + (
                                           kernel[1][1] * new_image[i][j]) + (
                                           kernel[1][2] * new_image[i][j + 1]) + (
                                           kernel[2][0] * new_image[i + 1][j - 1]) + (
                                           kernel[2][1] * new_image[i + 1][j]) + (
                                           kernel[2][2] * new_image[i + 1][j + 1])
    return output


def filter_moyenne(img, taille_h):
    X = copy.copy(img)
    image = copy.copy(img)
    i = int((taille_h) / 2)
    n = (taille_h ** 2)

    for k in range(i, image.shape[0] - i - 1):
        for l in range(i, image.shape[1] - i - 1):
            s = np.sum(image[k - i:k + i + 1, l - i:l + i + 1])
            X[k][l] = int(s / n)
    return X

def filter_median(img, taille_h):
    X = copy.copy(img)
    image = copy.copy(img)
    i = taille_h // 2
    for k in range(i, image.shape[0]-i-1):
        for l in range(i, image.shape[1]-i-1):
            X[k][l] = np.median(image[k-i:k+i+1, l-i:l+i+1])
    return X
