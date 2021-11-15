import cv2
import numpy as np
import numba
from skimage import exposure as ex
from im2dhist import im2dhist, imhist

class HE():
    def contrast_enhancement(self, img):
        if(len(img.shape)==2):      #gray
            outImg = ex.equalize_hist(img[:,:])*255 
        elif(len(img.shape)==3):    #RGB
            outImg = np.zeros((img.shape[0],img.shape[1],3))
            for channel in range(img.shape[2]):
                outImg[:, :, channel] = ex.equalize_hist(img[:, :, channel])*255

        outImg[outImg>255] = 255
        outImg[outImg<0] = 0
        return outImg.astype(np.uint8)

class HEQ():
    def __init__(self, w_neighboring=6):
        self.w_neighboring = w_neighboring

    def contrast_enhancement(self, img):
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        image_v = image_hsv[:, :, 2].copy()
        image_v_heq = self.im2dhisteq(image_v)

        image_hsv[:, :, 2] = image_v_heq.copy()
        result = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        return result

    def im2dhisteq(self, image):
        V = image.copy()
        [h, w] = V.shape
        V_hist = imhist(V)
        H_in = im2dhist(V, w_neighboring=self.w_neighboring)
        CDFx = np.cumsum(np.sum(H_in, axis=0)) # Kx1

        # normalizes CDFx
        CDFxn = (255*CDFx/CDFx[-1])

        PDFxn = np.zeros_like(CDFxn)
        PDFxn[0] = CDFxn[0]
        PDFxn[1:] = np.diff(CDFxn)

        X_transform = np.zeros((256))
        X_transform[np.where(V_hist > 0)] = PDFxn.copy()
        CDFxn_transform = np.cumsum(X_transform)

        bins = np.arange(256)
        image_equalized = np.floor(np.interp(V.flatten(), bins, CDFxn_transform).reshape(h, w)).astype(np.uint8)

        return image_equalized