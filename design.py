from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage

import cv2
from PIL import Image
import sys
import numpy as np

# import matplotlib.pyplot as plt


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('design.ui', self)
        self.show()

        ## connection between buttons and actions
        self.save.clicked.connect(self.savePhoto)
        self.open.clicked.connect(self.loadImage)
        self.binValueSlider.valueChanged['int'].connect(self.binValue)
        self.otsuButton.clicked.connect(self.otsu)
        self.histButton.clicked.connect(self.histogramme)
        self.resizeButton.clicked.connect(self.resizePic)


        self.filename = None  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display
        self.bin_value_now = 0 # will take the current binarisation value

    def loadImage(self):
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        print("this is file name : ", self.filename)
        if (self.filename):
            self.image = cv2.imread(self.filename)
            self.grayPicture = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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

    def binValue(self, value):
        print('Bin: ', value)
        if self.tmp is not None:
            self.setPhoto(self.manuel_bin(self.grayPicture, int(value)))
        else:
            print("ERROR in manuel_bin : image non trouvable")

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

