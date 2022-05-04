import copy
import math

from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage

import cv2
from PIL import Image
import sys
import numpy as np

# import matplotlib.pyplot as plt


def filt_gauss(sigma=0.1):
    matrix = np.array([
        [math.exp(- 2 / math.pow(sigma, 2)), math.exp(- 1 / (2 * math.pow(sigma, 2))),
         math.exp(- 2 / math.pow(sigma, 2))],
        [math.exp(- 1 / (2 * math.pow(sigma, 2))), 1, math.exp(- 1 / (2 * math.pow(sigma, 2)))],
        [math.exp(- 2 / math.pow(sigma, 2)), math.exp(- 1 / (2 * math.pow(sigma, 2))),
         math.exp(- 2 / math.pow(sigma, 2))],

    ])
    return (1 / (2 * (math.pow(sigma, 2)) * math.pi)) * matrix

class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('design.ui', self)
        self.show()

        ## connection between buttons and actions
        self.save.clicked.connect(self.savePhoto)
        self.open.clicked.connect(self.loadImage)
        self.binValueSlider.valueChanged['int'].connect(self.binValue)
        self.rotationSlider.valueChanged['int'].connect(self.rotate)
        self.otsuButton.clicked.connect(self.otsu)
        self.histButton.clicked.connect(self.histogramme)
        self.filtreMoyen.clicked.connect(self.filter_moyenne)
        self.medianFiltre.clicked.connect(self.filter_median)
        self.gaussFilter.clicked.connect(self.filter_gaussien)
        self.resetBtn.clicked.connect(self.reset)
        self.resizeButton.clicked.connect(self.resizePic)


        self.filename = None  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display
        self.bin_value_now = 0 # will take the current binarisation value


    def filter_gaussien(self):
        try:
            kernel = filt_gauss(0.1)

            image = copy.copy(self.grayPicture)
            new_image = np.zeros([image.shape[0] + 2, image.shape[1] + 2])

            new_image[1:-1, 1:-1] = image

            output = np.zeros([image.shape[0], image.shape[1]])

            for i in range(1, new_image.shape[0] - 1):
                for j in range(1, new_image.shape[1] - 1):
                    output[i - 1][j - 1] = (kernel[0][0] * new_image[i - 1][j - 1]) + (
                                kernel[0][1] * new_image[i - 1][j]) + (
                                                   kernel[0][2] * new_image[i - 1][j + 1]) + (
                                                       kernel[1][0] * new_image[i][j - 1]) + (
                                                   kernel[1][1] * new_image[i][j]) + (
                                                   kernel[1][2] * new_image[i][j + 1]) + (
                                                   kernel[2][0] * new_image[i + 1][j - 1]) + (
                                                   kernel[2][1] * new_image[i + 1][j]) + (
                                                   kernel[2][2] * new_image[i + 1][j + 1])
            self.setPhoto(output)
        except Exception as e:
            print(e)

    def reset(self):
        self.tmp = self.grayPicture
        self.setPhoto(self.grayPicture)

    def loadImage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        print("this is file name : ", self.filename)
        if (self.filename):
            self.image = cv2.imread(self.filename)
            self.tmp = self.grayPicture = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            dim = str(self.image.shape)
            self.width, self.height = Image.open(self.filename).size
            self.width = str(self.width)
            self.height = str(self.height)
            dimensiensLabel = dim + "(" + self.width + "x" + self.height + ")"
            self.setPhoto(self.image)

    def setPhoto(self, image):

        self.tmp = image
        #image = imutils.resize(image,width=320)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        try:
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
        except:
            print("Error in the set image place")


    def savePhoto(self):
        """ This function will save the image"""

        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        if (filename):
            cv2.imwrite(filename, self.tmp)
            print('Image saved as:', self.filename)

    def manuel_bin(self, img, seuil):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img[img > seuil] = 255
        img[img < seuil] = 0
        return img

    def rotate(self, value):
        # print('Rot: ', value)
        if self.tmp is not None:
            try:
                width, height = self.grayPicture.shape[0], self.grayPicture.shape[1]
                print(width)
                rotMat = cv2.getRotationMatrix2D((width/2+1, height/2+1), int(value), 1)
                rotImg = cv2.warpAffine(self.grayPicture, rotMat, (width, height))
                y_nonzero, x_nonzero = np.nonzero(rotImg)
                # rotImg = rotImg[np.min(y_nonzero):np.max(y_nonzero), np.min(x_nonzero):np.max(x_nonzero)]
                self.setPhoto(rotImg)
            except Exception as e:
                print(e)
        else:
            print("ERROR in manuel_bin : image not found!")

    def binValue(self, value):
        print('Bin: ', value)
        if self.tmp is not None:
            self.setPhoto(self.manuel_bin(self.grayPicture, int(value)))
        else:
            print("ERROR in manuel_bin : image not found!")

    def otsu(self):
        if self.tmp is not None:
            image = self.grayPicture
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

            self.setPhoto(self.manuel_bin(image, k_out))
        else:
            print("ERROR in otsu() : please select image first")

    def filter_moyenne(self, taille_h ):

        taille_h = 3
        X = copy.copy(self.grayPicture)
        image = copy.copy(self.grayPicture)
        i = int(taille_h / 2)
        n = (taille_h ** 2)
        print(taille_h)

        for k in range(i, image.shape[0] - i - 1):
            for l in range(i, image.shape[1] - i - 1):
                s = np.sum(image[k - i:k + i + 1, l - i:l + i + 1])
                X[k][l] = int(s / n)
        self.setPhoto(X)

    def filter_median(self, taille_h):
        taille_h = 3
        X = copy.copy(self.grayPicture)
        image = copy.copy(self.grayPicture)
        i = taille_h // 2
        for k in range(i, image.shape[0] - i - 1):
            for l in range(i, image.shape[1] - i - 1):
                X[k][l] = np.median(image[k - i:k + i + 1, l - i:l + i + 1])
        self.setPhoto(X)

    def histogramme(self):
        if self.tmp is not None:
            img = self.grayPicture
            img_hist = cv2.equalizeHist(img)
            self.setPhoto(img_hist)
        else:
            print("ERROR histogramme() : select an image please !!!")

    def resizePic(self):
        if self.tmp is not None:
            width = int(self.widthInput.text())
            height = int(self.heightInput.text())
            dim = (width, height)
            resized = cv2.resize(self.tmp, dim)
            self.setPhoto(resized)
            print("image resized to : ",dim)
        else:
            print("ERROR in resize() : image not exist")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui()
    app.exec()

