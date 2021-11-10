from scipy.fftpack import fft2
import scipy as N
from numpy import *
from numpy.fft import fft2
import numpy as np
import cv2
import os

class EPMP():
    def __init__(self, theta0 = 1.3945, theta1 = 9.1377, theta2 = 0.5):
        self.theta0 = theta0
        self.theta1 = theta1
        self.theta2 = theta2

    def contrast_enhancement(self, img):
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgH, imgS, imgV = cv2.split(imgHSV)
        T_HE = self.getTH(imgV)
        T_I = self.getTI()
        E = np.identity(256)
        DivGradient = abs(self.psf2otf([1, -1], [256, 256])) * abs(self.psf2otf([1, -1], [256, 256]))
        T = np.matrix((self.theta0*E + self.theta1*E + self.theta2*DivGradient)).getI()\
            *(np.reshape(np.matrix(self.theta0*T_HE.T + self.theta1 * T_I.T), (256, 1)))
        T = (T - min(T))/ (max(T) - min(T))
        imgHSV[:,:,2] = self.getTransform(imgV, T) * 255
        result = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
        return result

    def psf2otf(self, psf, dim):
        """ Based on the matlab function of the same name"""

        psf = np.reshape(psf,[1, len(psf)])
        h, w = np.shape(psf)

        mat = self.padzeros(psf, dim)

        mat = self.circshift(mat, 'horiz', -(w//2))
        mat = self.circshift(mat, 'vert', -(h//2))

        otf = fft2(mat)

        return otf

    @staticmethod
    def getTransform(img, T):
        hei, wid = img.shape
        img_res = np.reshape(np.array([T[int(i)] for i in img.flatten()]), (hei, wid))
        return img_res

    @staticmethod
    def getTI():
        return np.array([float(i) / 255 for i in range(0, 256)])

    @staticmethod
    def getTH(img):
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('float') / 255
        return cdf

    @staticmethod
    def padzeros(mat, dim):
        """ Pad the given matrix with zeros to the given dimensions
        Inputs:
        mat -- (scipy) matrix to pad
        dim -- final dimensions of (scipy) matrix to return
        Outputs:
        newmat -- matrix of dimensions dim. the contents of mat are in the
                upper left-hand corner
        """

        h, w = dim
        newmat = N.zeros((h, w))

        oldh, oldw = mat.shape

        if h < oldh or w < oldw:
            print("Size mismatch: (", h, ", ", w, ") < (", oldh, ", ", oldw, ")")
            # sys.exit(1)

        for i in range(min(oldh,h)):
            for j in range(min(oldw,w)):
                newmat[i, j] = mat[i, j]

        return newmat

    @staticmethod
    def circshift(mat, dir, shift):
        """ Shift the given matrix in the given direction by the given amount.
            Rows/columns get shifted around to the opposite side if they would
            otherwise "fall off" the end of the matrix.
            Horizontal movement is accomplished by postmultiplication of mat by
            an identity matrix shifted the same amount vertically.
            Vertical movement is accomplished by premultiplication of mat by an
            identity matrix shifted the same amount horizontally.
        Inputs:
        mat -- (scipy) matrix to shift
        dir -- direction (must be 'horiz' or 'vert')
        shift -- the amount to shift the matrix. no size limits are
                enforced, as the modulus is taken of the computed
                coordinate. negative values shift up or to the left; positive
                values shift down or to the right
        Outputs:
        newmat -- matrix shifted accordingly
        """

        h, w = mat.shape

        if dir == 'horiz':
            # identity must be w * w for resultant matrix to be h * w
            zeromat = N.zeros((w, w))

            for i in range(w):
                zeromat[i, (i + shift) % w] = 1

            newmat = dot(mat, zeromat)
            return newmat

        if dir == 'vert':
            # identity must be h * h for resultant matrix to be h * w
            zeromat = N.zeros((h, h))

            for i in range(h):
                zeromat[(i + shift) % h, i] = 1

            newmat = dot(zeromat, mat)
            return newmat

        return mat  # no direction specified
    