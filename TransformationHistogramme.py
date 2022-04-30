import copy
import numpy as np


def histogramme(img):
    image = copy.copy(img)
    index = []
    value = []
    min_val = image.min()
    max_val = image.max()
    for k in range(min_val, 255):
        count = image[image == k].size
        index.append(k)
        value.append(count)
    return np.array(value), np.array(index)


def prob_cum(img):
    output = {}
    image = copy.copy(img)
    somme = 0
    N = image.size
    histo_normaliser = histogramme(image)[0] / N
    im_histo_index = histogramme(image)[1]
    val_min = image.min()

    for k in im_histo_index:
        h = histo_normaliser[0]
        for i in range(0, k - val_min):
            h += histo_normaliser[i]
        output[k] = h
    return output


def egalisation(img):
    image = copy.copy(img)

    output = np.zeros([image.shape[0], image.shape[1]])
    _c = prob_cum(image)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            output[i][j] = _c[image[i][j]] * 255

    return output

def etirement(img):
    image = copy.copy(img)
    X = copy.copy(img)
    mx = np.amax(image)
    mn = np.amin(image)
    for i in range(0, image.shape[0]) :
        for j in range(0, image.shape[1]) :
            X[i][j] = (255/(mx - mn)*(img[i][j] - mn))
    return X

class Transformation:
    pass
