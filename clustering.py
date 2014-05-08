#!/usr/bin/env python
##
## clustering.py
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Fri Apr 25 18:16:06 2014 xs_yann
## Last update Thu May  8 22:41:41 2014 xs_yann
##

import os
import string
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

from clusterer import Clusterer

class Clustering:

    ESC = 27
    KEY_LEFT = 2
    KEY_RIGHT = 3
    PREVIEW_HEIGHT = 850

    def __init__(self, images):
        self.__index = 0
        clusterer = Clusterer()
        self.clusters = {}
        for image in images:
            self.clusters[image] = clusterer.getClusters(image)

    def __showClusters(self, index):
        for image, clusters in self.clusters.iteritems():
            index %= len(clusters)
            cluster = clusters[index]
            h, w = cluster.shape[:2]
            if h > self.PREVIEW_HEIGHT:
                factor = 1.0 / (float(h) / self.PREVIEW_HEIGHT)
                cluster = cv2.resize(cluster, (0, 0), fx=factor, fy=factor)
            cv2.imshow(os.path.basename(image), cluster)

    def run(self):
        self.__showClusters(0)
        while True:
            k = cv2.waitKey(0) & 0xFF

            if k == self.ESC:
                break
            elif chr(k) in string.digits:
                self.__index = int(chr(k))
                self.__showClusters(self.__index)
            elif k == self.KEY_LEFT:
                self.__index = max(0, self.__index - 1)
                self.__showClusters(self.__index)
            elif k == self.KEY_RIGHT:
                self.__index += 1
                self.__showClusters(self.__index)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Clustering")
    parser.add_argument("image", type=str, nargs="+", help="Input image")
    args = parser.parse_args()

    try:
        clustering = Clustering(args.image)
        clustering.run()
    except (OSError, cv2.error) as err:
        print err
        sys.exit(1)
