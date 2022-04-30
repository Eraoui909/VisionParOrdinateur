import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np


def manuel_bin(image, seuil):
    img = copy.copy(image)
    img[img > seuil] = 255
    img[img < seuil] = 0
    return img


def otsu(img):
    image = copy.copy(img)
    M = image.shape[0]
    N = image.shape[1]

    val_min = image.min()
    val_max = image.max()

    histo_normaliser = np.histogram(image, bins=range(val_min, val_max))[0] / (M * N)
    histo_normaliser_bins = np.histogram(image, bins=range(val_min, val_max))[1]

    variance_max = 0
    k_out = 0
    for k in histo_normaliser_bins:
        P1 = histo_normaliser[0]
        for i in range(0, k - val_min):
            P1 += histo_normaliser[i]
        P2 = 1 - P1

        m1 = 0
        for i in range(0, k - val_min):
            m1 += (1 / P1) * i * histo_normaliser[i]

        m2 = 0
        for i in range((k - val_min) + 1, histo_normaliser.shape[0]):
            m2 += (1 / P2) * i * histo_normaliser[i]

        mk = 0
        for i in range(0, k - val_min):
            mk += i * histo_normaliser[i]

        mg = 0
        for i in range(0, histo_normaliser.shape[0]):
            mg += i * histo_normaliser[i]

        variance = (np.power((mg * P1) - mk, 2)) / (P1 * P2)

        if variance > variance_max:
            variance_max = variance
            k_out = k

    return manuel_bin(image,k_out)


class Binarisation:
    pass
