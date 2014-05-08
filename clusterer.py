#!/usr/bin/env python
##
## clusterer.py
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Fri Apr 25 18:16:06 2014 xs_yann
## Last update Thu May  8 22:40:11 2014 xs_yann
##

import os
import time
import numpy as np
import cv2

class Clusterer:
    """
    Extract image clusters.
    """

    RANDOM = 0
    PLUSPLUS = 1
    K = 2 # TODO: Dynamic K

    __modes = { RANDOM: cv2.KMEANS_RANDOM_CENTERS, PLUSPLUS: cv2.KMEANS_PP_CENTERS }

    def getClusters(self, imagePath, backgroundColor=(0, 0, 0), mode=PLUSPLUS, verbose=True):
        """Returns clusters in image.
        """
        img = self.__readImage(imagePath)
        retval, labels, centers = self.__kmeans(img, mode, verbose)
        return self.__getClustersImages(img, labels, backgroundColor, verbose)

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

    def __kmeans(self, image, mode, verbose):
        """Apply K-Means algorithm on image.
        """
        samples = np.float32(image.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        if verbose:
            self.__startTime = time.time()
            print "K-Means computing..."

        return cv2.kmeans(samples, self.K, criteria, 10, self.__modes[mode])

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
