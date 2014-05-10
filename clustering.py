#!/usr/bin/env python
##
## clustering.py
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Fri Apr 25 18:16:06 2014 xs_yann
## Last update Sat May 10 17:52:03 2014 xs_yann
##

import os
import string
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt

from clusterer import Clusterer

class ClustererArgsParser:

    FEATURES_TABLE = { 'r': Clusterer.FEATURE_R,
                       'g': Clusterer.FEATURE_G,
                       'b': Clusterer.FEATURE_B,
                       'h': Clusterer.FEATURE_H,
                       's': Clusterer.FEATURE_S,
                       'v': Clusterer.FEATURE_V }

    def __init__(self, features):
        self.__features = features

    @property
    def features(self):
        """Convert "rgb" into FLAG_R | FLAG_G | FLAG_B.
        """
        flags = 0
        for f in self.__features:
            flags |= self.FEATURES_TABLE[f]
        return flags

class Clustering:

    ESC = 27
    KEY_LEFT = 2
    KEY_RIGHT = 3
    PREVIEW_HEIGHT = 850

    def __init__(self, images, features):
        self.__index = 0
        clusterer = Clusterer()
        self.clusters = {}

        argsParser = ClustererArgsParser(features)

        features = argsParser.features

        graph = [Clusterer.GRAPH_R, Clusterer.GRAPH_G]
        for image in images:
            self.clusters[image] = clusterer.getClusters(image,
                                                         mode=Clusterer.KMEANS_PP,
                                                         features=features,
                                                         graph=graph,
                                                         centers=True,
                                                         data=True)

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

def graph_arg(x):
    allowed = ClustererArgsParser.FEATURES_TABLE
    if not (1 <= len(x) <= 3) or sum([c in allowed for c in x]) != len(x):
        raise argparse.ArgumentTypeError("%s is not a graph arg" % x)
    return x

def features_arg(x):
    allowed = ClustererArgsParser.FEATURES_TABLE
    if sum([c in allowed for c in x]) != len(x):
        raise argparse.ArgumentTypeError("%s is not a features arg" % x)
    return x

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Clustering")
    parser.add_argument("image", type=str, nargs="+", help="Input image")
    parser.add_argument("-f", "--features", type=features_arg, default='rgb')
    parser.add_argument("-d", "--graph-data", type=graph_arg)
    parser.add_argument("-a", "--graph-all", type=graph_arg)
    parser.add_argument("-c", "--graph-centers", type=graph_arg)
    args = parser.parse_args()

    try:
        clustering = Clustering(args.image, args.features)
        clustering.run()
    except (OSError, cv2.error) as err:
        print err
        sys.exit(1)
