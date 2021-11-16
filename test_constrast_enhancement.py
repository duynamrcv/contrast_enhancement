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
    eff = EFF()   
    dhe = DHE()
    # epm = EPMP()
    # agc = AGC()
    # heq = HEQ()
    # dee = DEE()

    # paths = glob.glob("images/test1/*.JPG")
    # paths = glob.glob("test1/*")
    # for img_name in paths:
    st = time.time()
    img_name = "test1\\01.jfif"
    name = img_name.split('\\')[-1].split('.')[0]
    # img_name = sys.argv[1]
    # img_name = "images/test1/H30_0074_.JPG"
    img = cv2.imread(img_name)
    res_eff = eff.contrast_enhancement(img)
    res_dhe = dhe.contrast_enhancement(img)
    # res_epm = epm.contrast_enhancement(img)
    # res_agc = agc.contrast_enhancement(img)
    # res_heq = heq.contrast_enhancement(img)
    # res_dee = dee.contrast_enhancement(img)
    # result = dee.contrast_enhancement(img)

    cv2.imwrite("results/{}_eff.jpg".format(name), res_eff)
    cv2.imwrite("results/{}_dhe.jpg".format(name), res_dhe)
        # cv2.imwrite("results/{}_epm.jpg".format(name), res_epm)
        # cv2.imwrite("results/{}_agc.jpg".format(name), res_agc)
        # cv2.imwrite("results/{}_heq.jpg".format(name), res_heq)
        # cv2.imwrite("results/{}_dee.jpg".format(name), res_dee)
    en = time.time()
    print("Process time: {}s".format(en-st))

    # Show results
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.subplot(122)
    # plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.subplot(133)
    # plt.imshow(cv2.cvtColor(result_dhe, cv2.COLOR_BGR2RGB))
    # plt.show()

if __name__ == '__main__':
    main()