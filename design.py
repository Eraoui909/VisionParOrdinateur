import copy
import math
import os
import glob

from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage

import cv2
from PIL import Image
import sys
import numpy as np

from skimage.filters import sobel,gaussian
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt


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
        self.histButton.clicked.connect(self.histogram_traitement)
        self.filtreMoyen.clicked.connect(self.filter_moyenne)
        self.medianFiltre.clicked.connect(self.filter_median)
        self.gaussFilter.clicked.connect(self.filter_gaussien)
        self.resetBtn.clicked.connect(self.reset)
        self.resizeButton.clicked.connect(self.resizePic)
        self.negatifButton.clicked.connect(self.negative)
        self.egalisationButton.clicked.connect(self.egalisation)
        self.etirementButton.clicked.connect(self.etirement)
        self.croppButton.clicked.connect(self.cropping)
        self.erodeButton.clicked.connect(self.erosion)
        self.delateButton.clicked.connect(self.delatation)
        self.ouvertureButton.clicked.connect(self.ouverture)
        self.fermetureButton.clicked.connect(self.fermeture)
        self.hitOrMissButton.clicked.connect(self.hitOrMiss)
        self.kmeansButton.clicked.connect(self.kMeansSegmentation)
        self.croissRegButton.clicked.connect(self.croissanceRegions)
        self.repartitionRegiButton.clicked.connect(self.partition_traitememt)
        self.gradientButton.clicked.connect(self.contourGrad)
        self.prewittButton.clicked.connect(self.contourPrewit)
        self.sobelButton.clicked.connect(self.contourSobel)
        self.laplacienButton.clicked.connect(self.contourLaplacien)
        self.robertButton.clicked.connect(self.contourRobert)


        self.filename = None  # Will hold the image address location
        self.tmp = None  # Will hold the temporary image for display
        self.bin_value_now = 0 # will take the current binarisation value

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
        try:
            self.tmp = image
            #image = imutils.resize(image,width=320)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
        except Exception as ex:
            print("setPhoto() Exception : ",ex)


    def savePhoto(self):
        """ This function will save the image"""

        filename = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
        if (filename):
            cv2.imwrite(filename, self.tmp)
            print('Image saved as:', self.filename)

    def negative(self):
        if self.tmp is not None:
            img = self.grayPicture
            img = 1-img
            self.setPhoto(img)

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

    def filter_moyenne(self ):

        try:
            taille_h = int(self.moyenneInput.text())
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
        except Exception as e:
            print("Filter moyenne ERROR : ",e)

    def filter_median(self):
        try:
            taille_h = int(self.medianInput.text())
            X = copy.copy(self.grayPicture)
            image = copy.copy(self.grayPicture)
            i = taille_h // 2
            for k in range(i, image.shape[0] - i - 1):
                for l in range(i, image.shape[1] - i - 1):
                    X[k][l] = np.median(image[k - i:k + i + 1, l - i:l + i + 1])
            self.setPhoto(X)
        except Exception as e:
            print("Filter Median ERROR : ",e)

    def Histogramme(self):
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

    def filt_gauss(self, sigma=0.1):
        matrix = np.array([
            [math.exp(- 2 / math.pow(sigma, 2)), math.exp(- 1 / (2 * math.pow(sigma, 2))),
             math.exp(- 2 / math.pow(sigma, 2))],
            [math.exp(- 1 / (2 * math.pow(sigma, 2))), 1, math.exp(- 1 / (2 * math.pow(sigma, 2)))],
            [math.exp(- 2 / math.pow(sigma, 2)), math.exp(- 1 / (2 * math.pow(sigma, 2))),
             math.exp(- 2 / math.pow(sigma, 2))],

        ])
        return (1 / (2 * (math.pow(sigma, 2)) * math.pi)) * matrix

    def filter_gaussien(self):
        try:
            sigma = float(self.gaussienInput.text())
            print("sigma = ", sigma)
            kernel = self.filt_gauss(sigma)
            image = self.grayPicture
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

            ##output = Image.fromarray(output)
            self.setPhoto(output.astype(np.uint8))
        except Exception as e:
            print("Gaussien Filter ERROR : ", e)

    def histogramme(self,img):
        try:
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
        except Exception as e:
            print("Histogramme ERROR : ", e)

    def prob_cum(self, img):
        try:
            output = {}
            image = copy.copy(img)
            somme = 0
            N = image.size
            histo_normaliser = self.histogramme(image)[0] / N
            im_histo_index = self.histogramme(image)[1]
            val_min = image.min()

            for k in im_histo_index:
                h = histo_normaliser[0]
                for i in range(0, k - val_min):
                    h += histo_normaliser[i]
                output[k] = h
            return output
        except Exception as e:
            print("Prob_Cum ERROR : ", e)

    def egalisation(self):
        try:
            image = copy.copy(self.grayPicture)

            output = np.zeros([image.shape[0], image.shape[1]])
            _c = self.prob_cum(image)
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):
                    if image[i][j]<255:
                        output[i][j] = _c[image[i][j]] * 255


            self.setPhoto(output.astype(np.uint8))
        except Exception as e:
            print("Egalisation ERROR : ", e)

    def etirement(self):
        image = copy.copy(self.grayPicture)
        X = copy.copy(self.grayPicture)
        mx = np.amax(image)
        mn = np.amin(image)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                X[i][j] = (255 / (mx - mn) * (self.grayPicture[i][j] - mn))
        self.setPhoto(X)

    def cropping(self):
        try:
            cropx = int(self.cropXInput.text())
            cropy = int(self.cropYInput.text())
            img = copy.copy(self.grayPicture)
            y, x = img.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            self.setPhoto(img[starty:starty + cropy, startx:startx + cropx])
        except Exception as e:
            print("Cropping ERROR : ", e)

    def erosion(self):
        try:
            img = copy.copy(self.grayPicture)
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.erode(img, kernel)
            self.setPhoto(result)
        except Exception as e:
            print("erosion ERROR : ", e)

    def delatation(self):
        try:
            img = copy.copy(self.grayPicture)
            kernel = np.ones((5, 5), np.uint8)
            result = cv2.dilate(img, kernel)
            self.setPhoto(result)
        except Exception as e:
            print("delatation ERROR : ", e)

    def ouverture(self):
        try:
            img = copy.copy(self.grayPicture)
            kernel = np.ones((5, 5), np.uint8)
            erosion = cv2.erode(img, kernel)
            delatation = cv2.dilate(erosion, kernel)
            self.setPhoto(delatation)
        except Exception as e:
            print("delatation ERROR : ", e)

    def fermeture(self):
        try:
            img = copy.copy(self.grayPicture)
            kernel = np.ones((5, 5), np.uint8)
            delatation = cv2.dilate(img, kernel)
            erosion = cv2.erode(delatation, kernel)
            self.setPhoto(erosion)
        except Exception as e:
            print("delatation ERROR : ", e)

    def hitOrMiss(self):
        try:

            input_image = copy.copy(self.grayPicture)
            kernel = np.array((
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]), dtype="int")

            output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel)
            rate = 50
            kernel = (kernel + 1) * 127
            kernel = np.uint8(kernel)


            self.setPhoto(output_image)


        except Exception as e:
            print("hit or miss ERROR = ", e)

    def kMeansSegmentation(self):
        try:
            img = copy.copy(self.grayPicture)
            h, w = img.shape
            image_2d = img.reshape(h * w, 1)
            pixel_vals = np.float32(image_2d)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = int(self.nbrClusetInput.text())
            retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            segmented_data = centers[labels.flatten()]
            segmented_image = segmented_data.reshape((img.shape))

            self.setPhoto(segmented_image)

        except Exception as e:
            print("kMeansSegmentation ERROR : ", e)

    def Markers(self):

        img = self.image
        x, y, z = img.shape
        im_ = gaussian(img, sigma=4)
        if z == 3:
            br = sobel(im_[:, :, 0])
            bg = sobel(im_[:, :, 1])
            bb = sobel(im_[:, :, 2])
            brgb = br + bg + bb
        else:
            brgb = sobel(im_[:, :])

        markers = peak_local_max(brgb.max() - brgb)
        markers = peak_local_max(brgb.max() - brgb, threshold_rel=0.99, min_distance=50)
        return markers

    def croissanceRegions(self):

        try:
            markers = self.Markers()
            img = copy.copy(self.grayPicture)
            (thresh, bin_img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            h = img.shape[0]
            w = img.shape[1]

            out_img = np.zeros(shape=(img.shape), dtype=np.uint8)

            seeds = markers.tolist()
            for seed in seeds:
                x = seed[0]
                y = seed[1]
                out_img[x][y] = 255
            directs = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
            visited = np.zeros(shape=(img.shape), dtype=np.uint8)
            while len(seeds):
                seed = seeds.pop(0)
                x = seed[0]
                y = seed[1]
                visited[x][y] = 1

                for direct in directs:
                    cur_x = x + direct[0]
                    cur_y = y + direct[1]
                    if cur_x < 0 or cur_y < 0 or cur_x >= h or cur_y >= w:
                        continue
                    if (not visited[cur_x][cur_y]) and (bin_img[cur_x][cur_y] == bin_img[x][y]):
                        out_img[cur_x][cur_y] = 255
                        visited[cur_x][cur_y] = 1
                        seeds.append((cur_x, cur_y))
            bake_img = img.copy()
            h = bake_img.shape[0]
            w = bake_img.shape[1]
            for i in range(h):
                for j in range(w):
                    if out_img[i][j] != 255:
                        bake_img[i][j] = 0
                        bake_img[i][j] = 0
                        bake_img[i][j] = 0

            self.setPhoto(bake_img)
        except Exception as e:
            print("Croissance Regions = ",e)


    def contourGrad(self):

        try:
            img = copy.copy(self.grayPicture)
            (x, img) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

            Gx = copy.copy(img)
            Gy = copy.copy(img)

            result = copy.copy(img)
            [L, c] = img.shape

            for i in range(L - 1):
                for j in range(c - 1):
                    Gx[i][j] = img[i][j + 1] - img[i][j]
            for ii in range(L - 1):
                for jj in range(c - 1):
                    Gy[ii][jj] = img[ii + 1][jj] - img[ii][jj]
            for iii in range(L - 1):
                for jjj in range(c - 1):
                    result[iii][jjj] = math.sqrt(pow(Gx[iii][jjj], 2) + pow(Gy[iii][jjj], 2))
            self.setPhoto(result)

        except Exception as e:
            print("Contour Gradient ERROR = ", e)

    def contourPrewit(self):

        img = copy.copy(self.grayPicture)
        (x, img) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        Gx = img.copy()
        Gy = img.copy()
        result = img.copy()
        [L, c] = img.shape
        for i in range(L - 1):
            for j in range(c - 1):
                Gx[i][j] = img[i - 1][j - 1] + img[i - 1][j] + img[i - 1][j + 1] - img[i + 1][j - 1] - img[i + 1][j] - \
                           img[i + 1][j + 1]
                Gy[i][j] = -img[i - 1][j - 1] - img[i][j - 1] - img[i + 1][j - 1] + img[i - 1][j + 1] + img[i][j + 1] + \
                           img[i + 1][j + 1]
        for iii in range(L - 1):
            for jjj in range(c - 1):
                result[iii][jjj] = math.sqrt(pow(Gx[iii][jjj], 2) + pow(Gy[iii][jjj], 2))
        self.setPhoto(result)

    def contourSobel(self):
        img = copy.copy(self.grayPicture)
        (x,img) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        Gx = img.copy()
        Gy = img.copy()
        result = img.copy()
        [L, c] = img.shape
        for i in range(L - 1):
            for j in range(c - 1):
                Gx[i][j] = img[i-1][j - 1] + img[i-1][j]*2 + img[i-1][j + 1] - img[i + 1][j - 1] - img[i + 1][j]*2 - img[i + 1] \
                    [j + 1]
                Gy[i][j] = -img[i - 1][j - 1] - img[i][j - 1]*2 - img[i + 1][j - 1] + img[i - 1][j + 1] + img[i][j + 1]*2 + \
                           img[i + 1][j + 1]
        for iii in range(L - 1):
            for jjj in range(c - 1):
                result[iii][jjj] = math.sqrt(pow(Gx[iii][jjj], 2) + pow(Gy[iii][jjj], 2))

        self.setPhoto(result)

    def contourLaplacien(self):
        img = copy.copy(self.grayPicture)

        (x, img) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        result = img.copy()
        [L, c] = img.shape
        for i in range(L - 1):
            for j in range(c - 1):
                result[i][j] = img[i - 1][j] + img[i][j - 1] - img[i][j] * 4 + img[i][j + 1] + img[i + 1][j]

        self.setPhoto(result)

    def contourRobert(self):
        img = copy.copy(self.grayPicture)
        (x, img) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        X = copy.copy(img)
        Y = copy.copy(img)
        result = img.copy()

        for i in range(0, img.shape[0] - 1):
            for j in range(0, img.shape[1] - 1):
                X[i][j] = (X[i + 1][j] * -1) + (X[i][j + 1])
        for i in range(0, img.shape[0] - 1):
            for j in range(0, img.shape[1] - 1):
                Y[i][j] = (img[i + 1][j + 1] * -1) + (img[i][j])
        for iii in range(img.shape[0] - 1):
            for jjj in range(img.shape[1] - 1):
                result[iii][jjj] = math.sqrt(pow(X[iii][jjj], 2) + pow(Y[iii][jjj], 2))

        self.setPhoto(result)


    def Division_Judge(self, img, h0, w0, h, w):
        area = img[h0: h0 + h, w0: w0 + w]
        mean = np.mean(area)
        std = np.std(area, ddof=1)

        total_points = 0
        operated_points = 0

        for row in range(area.shape[0]):
            for col in range(area.shape[1]):
                if (area[row][col] - mean) < 2 * std:
                    operated_points += 1
                total_points += 1

        if operated_points / total_points >= 0.95:
            return True
        else:
            return False

    def Merge(self, img, h0, w0, h, w):

        for row in range(h0, h0 + h):
            for col in range(w0, w0 + w):
                if img[row, col] > 100 and img[row, col] < 200:
                    img[row, col] = 0
                else:
                    img[row, col] = 255

    def Recursion(self, img, h0, w0, h, w):
        # If the splitting conditions are met, continue to split
        if not self.Division_Judge(img, h0, w0, h, w) and min(h, w) > 5:
            # Recursion continues to determine whether it can continue to split
            # Top left square
            self.Division_Judge(img, h0, w0, int(h0 / 2), int(w0 / 2))
            # Upper right square
            self.Division_Judge(img, h0, w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
            # Lower left square
            self.Division_Judge(img, h0 + int(h0 / 2), w0, int(h0 / 2), int(w0 / 2))
            # Lower right square
            self.Division_Judge(img, h0 + int(h0 / 2), w0 + int(w0 / 2), int(h0 / 2), int(w0 / 2))
        else:
            # Merge
            self.Merge(img, h0, w0, h, w)

    def partition_traitememt(self):
        img_gray = copy.copy(self.grayPicture)

        segemented_img = img_gray.copy()
        self.Recursion(segemented_img, 0, 0, segemented_img.shape[0], segemented_img.shape[1])

        img_final = segemented_img
        self.setPhoto(img_final)

    def histogram_traitement(self):
        image = copy.copy(self.grayPicture)
        k = 0
        try:
            test = image.shape[2]
        except IndexError:
            k = 1
        if k == 1:
            h = self.histo(image)
            plt.subplot(1, 1, 1)
            plt.plot(h)
            plt.savefig('Y_X.png')
        else:
            for i in range(0, 3):
                h = self.histo(image[:, :, i])
                plt.subplot(1, 3, i + 1)
                plt.plot(h)
            plt.savefig('Y_X.png')

        img = cv2.imread("Y_X.png")
        self.img = img
        img_final = img
        files = glob.glob('Y_X.png')
        for i in files:
            os.remove(i)
        self.setPhoto(img_final)

    def histo(self, image):
        h = np.zeros(256)
        s = image.shape
        for j in range(s[0]):
            for i in range(s[1]):
                valeur = image[j, i]
                h[valeur] += 1
        return h

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui()
    app.exec()

