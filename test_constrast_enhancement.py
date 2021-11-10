import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import glob

from ExposureFusion import EFF
from DynamicHistogram import DHE, BPDHE

from HistogramEqualization import HE

def main():
    import time

    # Load Methods
    eff = EFF()
    dhe = DHE()

    st = time.time()
    # img_name = sys.argv[1]
    # img_name = "images/test1/H30_1320_.JPG"
    paths = glob.glob("images/test1/*.JPG")
    for i in range (10):
        name = paths[i].split("\\")[-1].split('.')[0]
        print(name)
        img = cv2.imread(paths[i])
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        result_eff = eff.contrast_enhancement(img)
        result_dhe = dhe.contrast_enhancement(img)
        en = time.time()
        print("Process time: {}s".format(en-st))

        # Save images
        cv2.imwrite("results/"+name+".JPG", img)
        cv2.imwrite("results/"+name+"_eff.JPG", result_eff)
        cv2.imwrite("results/"+name+"_dhe.JPG", result_dhe)

        # Show results
        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.subplot(132)
        # plt.imshow(cv2.cvtColor(result_eff, cv2.COLOR_BGR2RGB))
        # plt.subplot(133)
        # plt.imshow(cv2.cvtColor(result_dhe, cv2.COLOR_BGR2RGB))
        # plt.show()

if __name__ == '__main__':
    main()