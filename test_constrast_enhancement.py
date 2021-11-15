import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import glob

from ExposureFusion import EFF
from DynamicHistogram import DHE, BPDHE
from EntropyPreservingMapping import EPMP
from AdaptiveGammaCorrection import AGC
from HistogramEqualization import HE, HEQ
from DifferentEffectiveEnhancement import DEE

def main():
    import time

    # Load Methods
    # eff = EFF()   
    dhe = DHE()
    epm = EPMP()
    agc = AGC()
    heq = HEQ()
    dee = DEE()

    st = time.time()
    # img_name = sys.argv[1]
    img_name = "images/test1/H30_0074_.JPG"
    img = cv2.imread(img_name)
    result = dee.contrast_enhancement(img)
    en = time.time()
    print("Process time: {}s".format(en-st))

    # Show results
    plt.figure()
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.subplot(133)
    # plt.imshow(cv2.cvtColor(result_dhe, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    main()