import cv2
import matplotlib.pyplot as plt
from Binarisation import *
from TransformationHistogramme import *
import copy
from Filters import *

if __name__ == '__main__':
    url = "D:\MST_SDSI\S2\Vision par ordinateur\ImageTest\ImageTest\plage.jpg"
    image = cv2.imread(url)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    """

    plt.imshow(otsu(image), cmap='gray')
    plt.show()


    seuil = input("donner une seuil : ")
    seuil = int(seuil)
    img_bin2 = manuel_bin(image, seuil)
    print("img_bin2 = ", img_bin2)
    plt.imshow(img_bin2, cmap='gray')
    plt.show()

    cv2.equalizeHist()
    
    plt.imshow(image, cmap='gray')
    plt.show()

    plt.imshow(egalisation(image), cmap='gray')
    plt.show()

    plt.imshow(cv2.equalizeHist(image), cmap='gray')
    plt.show()
    
    plt.imshow(etirement(image), cmap='gray')
    plt.show()

    plt.hist(etirement(image).ravel(), 255, [0, 255])
    plt.show()
    
    
     # Generate Gaussian noise
    gauss = np.random.normal(0, 1, image.size)
    gauss = gauss.reshape(image.shape[0], image.shape[1]).astype('uint8')
    # Add the Gaussian noise to the image
    img_gauss = cv2.add(image, gauss)

    plt.imshow(img_gauss, cmap='gray')
    plt.show()

    plt.imshow(filter_gaussien(img_gauss, 2), cmap='gray')
    plt.show()

    output = []
    plt.imshow(cv2.GaussianBlur(img_gauss, (5, 5), 2), cmap='gray')
    plt.show()
    
    """



    plt.imshow(image, cmap='gray')
    plt.show()


    plt.imshow(filter_gaussien(image, 2), cmap='gray')
    plt.show()

    plt.imshow(cv2.GaussianBlur(image, (5, 5), 2), cmap='gray')
    plt.show()
