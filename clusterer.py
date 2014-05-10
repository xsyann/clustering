#!/usr/bin/env python
##
## clusterer.py
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Fri Apr 25 18:16:06 2014 xs_yann
## Last update Sat May 10 17:51:33 2014 xs_yann
##

import os
import time
import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Clusterer:
    """
    Extract image clusters.
    """

    KMEANS = 1
    KMEANS_PP = 2
    THRESHOLD = 3

    FEATURE_R = 1 << 0
    FEATURE_G = 1 << 1
    FEATURE_B = 1 << 2
    FEATURE_H = 1 << 3
    FEATURE_S = 1 << 4
    FEATURE_V = 1 << 5

    FEATURE_RGB = FEATURE_R | FEATURE_G | FEATURE_B

    GRAPH_R = 1
    GRAPH_G = 2
    GRAPH_B = 3
    GRAPH_H = 4
    GRAPH_S = 5
    GRAPH_V = 6

    GRAPH_RGB = [GRAPH_R, GRAPH_G, GRAPH_B]


    K = 3 # Cluster Count

    __modes = { KMEANS: cv2.KMEANS_RANDOM_CENTERS,
                KMEANS_PP: cv2.KMEANS_PP_CENTERS }

    # Private Static methods

    def __computeRGB(image):
        return np.float32(image.reshape((-1, 3)))

    def __hashRGB(flag, samples):
        table = { Clusterer.FEATURE_R: samples[:,2:3],
                  Clusterer.FEATURE_G: samples[:,1:2],
                  Clusterer.FEATURE_B: samples[:,0:1] }
        return table[flag]

    def __computeHSV(image):
        return np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape((-1, 3)))

    def __hashHSV(flag, samples):
        table = { Clusterer.FEATURE_H: samples[:,0:1],
                  Clusterer.FEATURE_S: samples[:,1:2],
                  Clusterer.FEATURE_V: samples[:,2:3] }
        return table[flag]

    # Features

    __featuresTable = [((FEATURE_R, FEATURE_G, FEATURE_B), __computeRGB, __hashRGB),
                       ((FEATURE_H, FEATURE_S, FEATURE_V), __computeHSV, __hashHSV) ]

    # Public methods

    def getClusters(self, imagePath, mode=KMEANS_PP, features=FEATURE_RGB,
                    graph=GRAPH_RGB, centers=True, data=True,
                    backgroundColor=(0, 0, 0), verbose=True):
        """Returns clusters in image.
        """
        img = self.__readImage(imagePath)
        if mode == self.THRESHOLD:
            labels = self.__threshold(img, verbose)
        else:
            retval, labels, centers = self.__kmeans(img, mode, features, verbose)
        return self.__getClustersImages(img, labels, backgroundColor, verbose)

    # Private methods

    def __getClustersImages(self, image, labels, backgroundColor, verbose):
        """Create clusters images from k-means results.
        """
        clusters = []
        for i in xrange(self.K):
            if verbose:
                print "Cluster {}...".format(i)
            clusters.append(self.__getClusterImage(image, labels, i, backgroundColor))
        if verbose:
            print "Calculation time: {:.3f} s".format(time.time() - self.__startTime)
        return clusters

    def __getClusterImage(self, image, labels, clusterLabel, bgColor):
        """Create a cluster image from k-means results.
        """
        cluster = np.copy(np.copy(image).reshape((-1, 3)))
        cond = (labels.flatten() == clusterLabel)
        cluster = np.uint8(np.where(zip(cond, cond, cond), cluster, bgColor))

        return cluster.reshape(image.shape)

    def __threshold(self, image, verbose):
        if verbose:
            self.__startTime = time.time()
            print "Threshold computing..."
        flatten = np.copy(np.copy(image).reshape((-1, 3)))
#        gray = np.uint8(np.sum(flatten, axis=1) / 3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape((-1, 3))
        hsv = hsv[:,0]
#        range = 255 / self.K
        range = 180 / self.K
        labels = np.zeros_like(hsv)
        for i in xrange(1, self.K):
            labels += (hsv > (range * i))
            #labels += (gray > (range * i))

        return labels

    def __getFeatures(self, image, flags):
        """Extract features from features flags and stack them in an array.
        """
        samples = []
        for components, compute, hash in self.__featuresTable:
            if self.__hasComponent(flags, components):
                samples.append((components, compute(image), hash))

        features = None
        for i in xrange(flags.bit_length()):
            flag = ((flags >> i) & 1) << i
            if flag:
                for components, sample, hash in samples:
                    if flag in components:
                        features = hash(flag, sample) if features is None else \
                                 np.hstack((features, hash(flag, sample)))
        return features

    def __hasComponent(self, flags, components):
        """Return True if at least one item in components is in flags.
        """
        return sum([flags & f for f in components]) > 0

    def __kmeans(self, image, mode, features, verbose):
        """Apply K-Means algorithm on image.
        """
        if verbose:
            self.__startTime = time.time()
            print "K-Means computing..."

        samples = self.__getFeatures(image, features)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        retval, labels, centers = cv2.kmeans(samples, self.K,
                                             criteria, 10, self.__modes[mode])

        return retval, labels, centers

    def __readImage(self, path):
        """Load image from path.
        Raises OSError exception if path doesn't exist or is not an image.
        """
        if not os.path.isfile(path):
            raise OSError(2, 'File not found', path)
        img = cv2.imread(path)
        if img is None:
            raise OSError(2, 'File not an image', path)
        return img
