import cv2
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt

class AGC():
    def __init__(self, threshold = 0.3):
        self.threshold = threshold

    def contrast_enhancement(self, img):
        # Extract intensity component of the image
        YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y = YCrCb[:,:,0]
        # Determine whether image is bright or dimmed
        exp_in = 112 # Expected global average intensity 
        M,N = img.shape[:2]
        mean_in = np.sum(Y/(M*N)) 
        t = (mean_in - exp_in)/ exp_in
        
        # Process image for gamma correction
        img_output = None
        if t < -self.threshold: # Dimmed Image
            result = self.process_dimmed(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        elif t > self.threshold:
            result = self.process_bright(Y)
            YCrCb[:,:,0] = result
            img_output = cv2.cvtColor(YCrCb,cv2.COLOR_YCrCb2BGR)
        else:
            img_output = img
        return img_output

    def process_bright(self, img):
        img_negative = 255 - img
        agcwd = self.image_agcwd(img_negative, a=0.25, truncated_cdf=False)
        reversed = 255 - agcwd
        return reversed

    def process_dimmed(self, img):
        agcwd = self.image_agcwd(img, a=0.75, truncated_cdf=True)
        return agcwd

    @staticmethod
    def image_agcwd(img, a=0.25, truncated_cdf=False):
        h,w = img.shape[:2]
        hist,bins = np.histogram(img.flatten(),256,[0,256])
        # cdf = hist.cumsum()
        prob_normalized = hist / hist.sum()

        unique_intensity = np.unique(img)
        prob_min = prob_normalized.min()
        prob_max = prob_normalized.max()
        
        pn_temp = (prob_normalized - prob_min) / (prob_max - prob_min)
        pn_temp[pn_temp>0] = prob_max * (pn_temp[pn_temp>0]**a)
        pn_temp[pn_temp<0] = prob_max * (-((-pn_temp[pn_temp<0])**a))
        prob_normalized_wd = pn_temp / pn_temp.sum() # normalize to [0,1]
        cdf_prob_normalized_wd = prob_normalized_wd.cumsum()
        
        if truncated_cdf: 
            inverse_cdf = np.maximum(0.5,1 - cdf_prob_normalized_wd)
        else:
            inverse_cdf = 1 - cdf_prob_normalized_wd
        
        img_new = img.copy()
        for i in unique_intensity:
            img_new[img==i] = np.round(255 * (i / 255)**inverse_cdf[i])
    
        return img_new