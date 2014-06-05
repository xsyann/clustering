#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
## clusterer.py
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Fri Apr 25 18:16:06 2014 xs_yann
## Last update Thu Jun  5 13:52:03 2014 xs_yann
##

import os
import time
import numpy as np
import cv2
import matplotlib
import urllib2
from voronoi import voronoi
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Graph:

    def __init__(self):
        self.__points = []
        self.kmode = 0

    def addElbow(self, point):
        self.__points.append((point, d, h))

    def add(self, point):
        self.__points.append(point)

    def addGoal(self, point):
        self.__goal = point

    def addClosest(self, point):
        self.__closest = point

    def addData(self, img, samples, labels, centers, orderedFeatures):
        self.__img = img
        self.__samples = samples
        self.__labels = labels
        self.__centers = centers
        self.__orderedFeatures = orderedFeatures

    def graph(self, figure, verbose):
        figure.clf()
        dataSubplot = 111

        kmodes = { Clusterer.KMODE_ELBOW: self.__graphElbow,
                   Clusterer.KMODE_GAPSTAT: self.__graphGapStat,
                   Clusterer.KMODE_THRESHOLD: self.__graphThreshold,
                   Clusterer.KMODE_GAPTHRESHOLD: self.__graphGapThreshold,
                   Clusterer.KMODE_SLIDER: self.__graphSlider,
                   Clusterer.KMODE_FK: self.__graphFK }

        if self.kmode in kmodes:
            dataSubplot = kmodes[self.kmode](figure) # Graph K estimation

        self.__graphData(figure, self.__img, self.__samples,
                         self.__labels, self.__centers, self.__orderedFeatures,
                         dataSubplot, verbose)

    def __graphData(self, figure, img, samples, labels, centers, orderedFeatures,
                    subplot, verbose):
        ndim = len(orderedFeatures)
        if not 1 <= ndim <= 3:
            return
        clustersLabel = np.unique(labels)
        rgbSamples = img.reshape((-1, 3))
        ax = figure.add_subplot(subplot) if ndim <= 2 else figure.add_subplot(subplot, projection="3d")
        for clusterLabel in clustersLabel:
            if verbose:
                print "Graph cluster {}...".format(clusterLabel)
            cluster = samples[labels.ravel() == clusterLabel]
            color = rgbSamples[labels.ravel() == clusterLabel] / 255.
            color[:,[0, 2]] = color[:,[2, 0]] # BGR to RGB
            length = cluster.shape[0]
            limit = 8000 if ndim == 2 else 800
            limit *= 1. / ndim
            keep = np.random.permutation(length)[:limit]
            cluster = cluster[keep]
            color = color[keep]
            if ndim == 2:
                ax.scatter(cluster[:,0], cluster[:,1], c=color, lw=0)
            elif ndim == 1:
                ax.hist(cluster[:,0], color=color[0])
            else:
                ax.scatter(cluster[:,0], cluster[:,1], cluster[:,2], c=color, lw=0)
        if ndim == 2:
            ax.scatter(centers[:,0], centers[:,1], c='y', marker='s', lw=1)
            segments = voronoi(centers)
            lines = matplotlib.collections.LineCollection(segments, color='k')
            ax.add_collection(lines)
        elif ndim == 3:
            ax.scatter(centers[:,0], centers[:,1], centers[:,2], s=80, c='y', marker='s', lw=1)
            ax.set_zlabel(Clusterer.getFeatureName(orderedFeatures[2]))
        if ndim == 1:
            ax.set_ylabel('Reduced samples')
        if 2 <= ndim <= 3:
            ax.set_ylabel(Clusterer.getFeatureName(orderedFeatures[1]))
        ax.set_xlabel(Clusterer.getFeatureName(orderedFeatures[0]))

    def __graphFK(self, figure):
        if not self.__points:
            return
        ax = figure.add_subplot(121)
        ks, fs = zip(*self.__points)
        ax.plot(ks, fs, linestyle='-', marker='o', c='b')
        ax.axhline(y=.85, c='r')
        ax.set_xlabel("Cluster count")
        ax.set_ylabel("f(K)")
        return 122

    def __graphGapStat(self, figure):
        if not self.__points:
            return
        ax = figure.add_subplot(221)
        ks, wks, ewks, sks, gaps, delta = zip(*self.__points)
        ax.plot(ks, wks, linestyle='-', marker='o', c='b', label='observed')
        ax.plot(ks, ewks, linestyle='-', marker='o', c='r', label='estimated')
        ax.legend()
        ax.set_xlabel("Cluster count")
        ax.set_ylabel("Compactness")

        bx = figure.add_subplot(223)

        bx.plot(ks, gaps, linestyle='-', marker='o', c='b', label='gaps')
        bx.bar(ks, delta, align='center', alpha=0.5, color='g', label='delta')
        bx.plot(ks, sks, linestyle='-', marker='o', c='y', label='error')
        bx.set_xlabel("Cluster count")
        bx.set_ylabel("Gap")
        return 122

    def __graphThreshold(self, figure):
        if not self.__points:
            return
        ax = figure.add_subplot(121)
        ks, ts = zip(*(self.__points))
        ax.plot(ks, ts, linestyle='-', marker='o', c='g')
        ax.axhline(y=Clusterer.THRESHOLD, c='r')
        plt.xlabel("Cluster count")
        plt.ylabel("Compactness")
        return 122

    def __graphSlider(self, figure):
        if not self.__points:
            return
        ax = figure.add_subplot(221)
        ax.axis("equal")
        ks, ts, dists = zip(*(self.__points))
        ax.plot((0, 1), (1, 0), linestyle='-', marker='o', c='r')
        ax.plot(ks, ts, linestyle='-', marker='o', c='g')
        ax.plot([self.__closest[0], self.__goal[0]], [self.__closest[1], self.__goal[1]],
                linestyle='-', marker='o', c='b')
        ax.scatter(self.__goal[0], self.__goal[1], linestyle='-', marker='o', s=60, c='r')
        plt.xlabel("Cluster count rate")
        plt.ylabel("Compactness")

        bx = figure.add_subplot(223)
        ks = [(k * (Clusterer.K_MAX - Clusterer.K_MIN)) + Clusterer.K_MIN for k in ks]
        bx.bar(ks, dists, alpha=.5, width=0.4, color='g', align='center')
        best = min(dists[1:])
        bx.axhline(y=best, c='r')
        plt.xlabel("Cluster count")
        plt.ylabel("Distance from goal")
        return 122

    def __graphGapThreshold(self, figure):
        if not self.__points:
            return
        ax = figure.add_subplot(121)
        ks, gaps, comps = zip(*(self.__points))
        ax.plot(ks, comps, linestyle='-', marker='o', c='b')
        ax.bar(ks, gaps, align='center', alpha=0.5, color='g', label='gap')
        ax.axhline(y=Clusterer.GAP_THRESHOLD, c='r')
        ax.legend()
        plt.xlabel("Cluster count")
        plt.ylabel("Compactness")
        return 122

    def __graphElbow(self, figure):
        if not self.__points:
            return
        pFirst = self.__points[0][0]
        pLast = self.__points[-1][0]
        ax = figure.add_subplot(121)
        ax.axis("equal")
        ax.plot([pFirst[0], pLast[0]], [pFirst[1], pLast[1]], linestyle='-', marker='o', c='g')
        compactness = []
        ks, cs = [], []
        for pCurve, d, h in self.__points:
            k, c = pCurve
            ks.append(k)
            cs.append(c)
            if h is not None:
                ax.plot([h[0], pCurve[0]], [h[1], pCurve[1]], linestyle='-', marker='o', c='r')
        ax.plot(ks, cs, linestyle='-', marker='o', c='b')
        plt.xlabel("Cluster count")
        plt.ylabel("Compactness")
        return 122

class Clusterer:
    """
    Extract image clusters.
    """

    ########################################################
    # Public Static attributes

    KMEANS = 1
    KMEANS_PP = 2

    FEATURE_R = 1 << 0
    FEATURE_G = 1 << 1
    FEATURE_B = 1 << 2
    FEATURE_H = 1 << 3
    FEATURE_S = 1 << 4
    FEATURE_V = 1 << 5
    FEATURE_L = 1 << 6
    FEATURE_GR = 1 << 7
    FEATURE_CIEX = 1 << 8
    FEATURE_CIEY = 1 << 9
    FEATURE_CIEZ = 1 << 10
    FEATURE_C = 1 << 11
    FEATURE_M = 1 << 12
    FEATURE_Y = 1 << 13
    FEATURE_K = 1 << 14

    # Features unordered (flags)
    FEATURES_RGB = FEATURE_R | FEATURE_G | FEATURE_B
    FEATURES_HSV = FEATURE_H | FEATURE_S | FEATURE_V

    # K Modes

    KMODE_ELBOW = 1
    KMODE_THRESHOLD = 2
    KMODE_GAPTHRESHOLD = 3
    KMODE_GAPSTAT = 4
    KMODE_FK = 5
    KMODE_SLIDER = 6
    KMODE_USER = 7

    K_MIN = 1 # Minmum cluster count
    K_MAX = 12 # Maxium cluster count

    K_LAST_ELBOW = 20
    ALGO_OPTI = False

    THRESHOLD = 0.093
    GAP_THRESHOLD = 0.093


    ########################################################
    # Public static methods

    @staticmethod
    def getDefaultFeatures():
        """Return default features.
        """
        return Clusterer.FEATURES_RGB

    @staticmethod
    def getAllFeatures():
        """Return all features.
        """
        features = 0
        for components, names, compute, table in Clusterer.__featuresTable:
            for feature in components:
                features |= feature
        return features

    @staticmethod
    def getFeatureName(feature):
        """Return the name of 'feature'.
        """
        for components, names, compute, table in Clusterer.__featuresTable:
            if feature in components:
                return names[components.index(feature)]
        return ""

    @staticmethod
    def getFeatureByName(name):
        """Return feature named 'name'.
        """
        for components, names, compute, table in Clusterer.__featuresTable:
            if name in names:
                return components[names.index(name)]
        return 0

    @staticmethod
    def getDefaultMode():
        """Return default mode.
        """
        return Clusterer.KMEANS_PP

    @staticmethod
    def getAllModes():
        """Return all features.
        """
        return ([(mode, name) for mode, name in Clusterer.__modeNames.iteritems()])

    @staticmethod
    def getModeName(mode):
        """Return the name of 'mode'.
        """
        if name in Clusterer.__modeNames:
            return Clusterer.__modeNames[name]
        return ""

    @staticmethod
    def getModeByName(name):
        """Return mode named 'name'.
        """
        for mode, modeName in Clusterer.__modeNames.iteritems():
            if name == modeName:
                return mode
        return 0

    @staticmethod
    def getDefaultKMode():
        """Return default mode.
        """
        return Clusterer.KMODE_GAPSTAT

    @staticmethod
    def getAllKModes():
        """Return all features.
        """
        return ([(mode, name) for mode, name in Clusterer.__kmodeNames.iteritems()])


    @staticmethod
    def getKModeByName(name):
        """Return mode named 'name'.
        """
        for mode, modeName in Clusterer.__kmodeNames.iteritems():
            if name == modeName:
                return mode
        return 0

    @staticmethod
    def readImage(path):
        """Load image from path.
        Raises OSError exception if path doesn't exist or is not an image.
        """
        img = None
        if not os.path.isfile(path):
            try:
                req = urllib2.urlopen(path)
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.CV_LOAD_IMAGE_COLOR)
            except ValueError:
                raise OSError(2, 'File not found', path)
        else:
            img = cv2.imread(path)
        if img is None:
            raise OSError(2, 'File not recognized', path)
        return img


    ########################################################
    # Private static methods

    # RGB
    def __computeRGB(image):
        return np.float32(image.reshape((-1, 3)))

    def __tableRGB(flag, samples):
        table = { Clusterer.FEATURE_R: samples[:,2:3],
                  Clusterer.FEATURE_G: samples[:,1:2],
                  Clusterer.FEATURE_B: samples[:,0:1] }
        return table[flag]

    # HSV
    def __computeHSV(image):
        return np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape((-1, 3)))

    def __tableHSV(flag, samples):
        table = { Clusterer.FEATURE_H: samples[:,0:1],
                  Clusterer.FEATURE_S: samples[:,1:2],
                  Clusterer.FEATURE_V: samples[:,2:3] }
        return table[flag]

    # HLS
    def __computeHLS(image):
        return np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2HLS).reshape((-1, 3)))

    def __tableHLS(flag, samples):
        table = { Clusterer.FEATURE_H: samples[:,0:1],
                  Clusterer.FEATURE_L: samples[:,1:2],
                  Clusterer.FEATURE_S: samples[:,2:3] }
        return table[flag]

    # Gray
    def __computeGray(image):
        return np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape((-1, 1)))

    def __tableGray(flag, samples):
        table = { Clusterer.FEATURE_GR: samples[:] }
        return table[flag]

    # CIE XYZ
    def __computeCXYZ(image):
        return np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2XYZ).reshape((-1, 3)))

    def __tableCXYZ(flag, samples):
        table = { Clusterer.FEATURE_CIEX: samples[:,0:1],
                  Clusterer.FEATURE_CIEY: samples[:,1:2],
                  Clusterer.FEATURE_CIEZ: samples[:,2:3] }
        return table[flag]

    # CMYK
    def __computeCMYK(image):
        def bgr2cmyk(bgr):
            cmykScale = 100
            r, g, b = bgr[2], bgr[1], bgr[0]
            if (r == 0) and (g == 0) and (b == 0):
                return 0, 0, 0, cmykScale # black

            # rgb [0,255] -> cmy [0,1]
            c = 1 - r / 255.
            m = 1 - g / 255.
            y = 1 - b / 255.

            # extract out k [0,1]
            min_cmy = min(c, m, y)
            c = (c - min_cmy) / (1 - min_cmy)
            m = (m - min_cmy) / (1 - min_cmy)
            y = (y - min_cmy) / (1 - min_cmy)
            k = min_cmy

            # rescale to the range [0,cmyk_scale]
            return (c * cmykScale, m * cmykScale, y * cmykScale, k * cmykScale)
        print "RGB to CMYK..."
        bgr = np.float32(image.reshape((-1, 3)))
        cmyk = np.float32(np.apply_along_axis(bgr2cmyk, 1, bgr))
        return cmyk

    def __tableCMYK(flag, samples):
        table = { Clusterer.FEATURE_C: samples[:,0:1],
                  Clusterer.FEATURE_M: samples[:,1:2],
                  Clusterer.FEATURE_Y: samples[:,2:3],
                  Clusterer.FEATURE_K: samples[:,3:4] }
        return table[flag]


    ########################################################
    # Private Static attributes

    __featuresTable = [((FEATURE_R, FEATURE_G, FEATURE_B), ('Red', 'Green', 'Blue'),
                        __computeRGB, __tableRGB),
                       ((FEATURE_H, FEATURE_S, FEATURE_V), ('Hue', 'Saturation', 'Value'),
                        __computeHSV, __tableHSV),
                       ((FEATURE_H, FEATURE_L, FEATURE_S), ('Hue', 'Luminosity', 'Saturation'),
                        __computeHSV, __tableHLS),
                       ((FEATURE_GR,), ('Gray',), __computeGray, __tableGray),
                       ((FEATURE_CIEX, FEATURE_CIEY, FEATURE_CIEZ), ('CIE X', 'CIE Y', 'CIE Z'),
                        __computeCXYZ, __tableCXYZ),
                       ((FEATURE_C, FEATURE_M, FEATURE_Y, FEATURE_K), ('Cyan', 'Magenta', 'Yellow', 'Black'),
                        __computeCMYK, __tableCMYK) ]

    __modeNames = { KMEANS: "K-Means",
                    KMEANS_PP: "K-Means++" }

    __kmodeNames = { KMODE_ELBOW: "Elbow",
                     KMODE_THRESHOLD: "Threshold",
                     KMODE_GAPTHRESHOLD: "Gap Threshold",
                     KMODE_GAPSTAT: "Gap Statistic",
                     KMODE_FK: "f(K)",
                     KMODE_SLIDER: "Slider",
                     KMODE_USER: "Custom" }


    __kmeansModes = { KMEANS: cv2.KMEANS_RANDOM_CENTERS,
                      KMEANS_PP: cv2.KMEANS_PP_CENTERS }


    ########################################################
    # Public methods

    def __init__(self):
        self.__startTime = None
        self.__orderedFeatures = []
        self.__graph = Graph()

    def graph(self, figure, verbose=True):
        self.__graph.graph(figure, verbose)

    def getClusters(self, imagePath, clusterCount=0, mode=KMEANS_PP, kmode=KMODE_FK, features=FEATURES_RGB,
                    backgroundColor=(0, 0, 0), verbose=True, slider=None):
        """Returns clusters in image (clusterCount = 0 for automatic cluster count).
        """
        img = Clusterer.readImage(imagePath)
        self.__startTime = time.time()
        self.__graph.kmode = kmode
        self.__slider = slider
        if mode == self.KMEANS or mode == self.KMEANS_PP:
            if verbose:
                print "K-Means computing..."
            compactness, labels, centers = self.__kmeans(img, clusterCount, mode,
                                                         kmode, features, verbose)
        return self.__getClustersImages(img, labels, backgroundColor, verbose)


    ########################################################
    # Private methods

    # Clusters

    def __getClustersImages(self, image, labels, backgroundColor, verbose):
        """Create clusters images from cluster labels.
        """
        clusters = []
        for i in xrange(len(np.unique(labels))):
            if verbose:
                print "Creating Cluster {}...".format(i)
            clusters.append(self.__getClusterImage(image, labels, i, backgroundColor))
        if verbose:
            print "Total calculation time: {:.3f} s".format(time.time() - self.__startTime)
        return clusters

    def __getClusterImage(self, image, labels, clusterLabel, bgColor):
        """Create a cluster image from a cluster label.
        """
        h, w = image.shape[:2]
        imageRGBA = cv2.cvtColor(image, cv2.cv.CV_BGR2BGRA)
        cluster = np.copy(np.copy(imageRGBA).reshape((-1, 4)))
        bg = np.zeros(cluster.shape, np.uint8)
        bg[:,:] = bgColor + (255,) if bgColor else (0, 0, 0, 0)
        cond = (labels.flatten() == clusterLabel)

        cluster = np.uint8(np.where(zip(cond, cond, cond, cond), cluster, bg))

        return cluster.reshape(imageRGBA.shape)

    # Algorithms

    def __linearEquation(self, p1, p2):
        """Return the tuple (m, p) of the linear equation y = mx + p
        of the line formed by distinct points p1 and p2."""
        vecDir = (p2[0] - p1[0] , p2[1] - p1[1])
        m = vecDir[1] / vecDir[0]
        p = (p1[1] - (m * p1[0]))

        return (m, p)

    def __distance(self, p1, p2):
        """Return the distance between p1 and p2.
        """
        return np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2))

    def __computeKmeans(self, samples, k, mode, verbose):
        """Compute k-means with k clusters.
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        if verbose:
            print "K-Means for {} Clusters...".format(k),
            startTime = time.time()
        compactness, labels, centers = cv2.kmeans(samples, k,
                                             criteria, 2, self.__kmeansModes[mode])
        if verbose:
                print "in {:.3f} s".format(time.time() - startTime)
        return compactness, labels, centers

    def __reduceCompactness(self, compactness, baseValue):
        """Reduce the compactness in the range [K_MIN, K_MAX].
        """
        baseValue = np.float64(baseValue)
        compactness = np.float64(compactness) / baseValue * (self.K_MAX - self.K_MIN) + self.K_MIN
        return compactness

    def __rangeCompactness(self, compactness):
        """Reduce the compactness from [K_MIN, K_MAX] to the range [0, 1].
        """
        return (compactness - self.K_MIN) / (self.K_MAX - self.K_MIN)

    def __estimateKThreshold(self, samples, mode, verbose):
        """Estimate cluster count using threshold.
        """
        firstCompactness = self.__computeKmeans(samples, self.K_MIN, mode, verbose)[0]
        baseCompactness = firstCompactness
        firstCompactness = self.__rangeCompactness(self.__reduceCompactness(firstCompactness, baseCompactness))

        pFirst = self.K_MIN, firstCompactness

        self.__graph.add(pFirst)
        bestK = 0

        for k in xrange(self.K_MIN + 1, self.K_MAX + 1):
            xCompactness = self.__computeKmeans(samples, k, mode, verbose)[0]
            xCompactness = self.__rangeCompactness(self.__reduceCompactness(xCompactness, baseCompactness))
            pCurve = (k, xCompactness)
            self.__graph.add(pCurve)
            if xCompactness < self.THRESHOLD and bestK == 0:
                bestK = k
                if self.ALGO_OPTI:
                    break
        return bestK

    def __estimateKSlider(self, samples, mode, verbose):
        """Estimate cluster count using threshold.
        """
        baseCompactness = 0
        bestK = 0

        pGoal = (self.__slider / 100.0, (100 - self.__slider) / 100.0)
        self.__graph.addGoal(pGoal)

        dists = []

        for k in xrange(self.K_MIN, self.K_MAX + 1):
            xCompactness = self.__computeKmeans(samples, k, mode, verbose)[0]
            if baseCompactness == 0:
                baseCompactness = xCompactness
            xCompactness = self.__rangeCompactness(self.__reduceCompactness(xCompactness, baseCompactness))
            pCurve = (self.__rangeCompactness(np.float64(k)), xCompactness)
            dist = self.__distance(pCurve, pGoal)
            self.__graph.add((pCurve[0], pCurve[1], dist))
            dists.append((dist, k, pCurve))
        best = min(dists[1:])
        self.__graph.addClosest(best[2])
        return best[1]

    def __estimateKGapThreshold(self, samples, mode, verbose):
        """Estimate cluster count using gap threshold.
        """
        firstCompactness = self.__computeKmeans(samples, self.K_MIN, mode, verbose)[0]
        baseCompactness = firstCompactness
        firstCompactness = self.__rangeCompactness(self.__reduceCompactness(firstCompactness, baseCompactness))

        pFirst = self.K_MIN, firstCompactness

        bestK = 0
        prevCompactness = firstCompactness

        for k in xrange(self.K_MIN + 1, self.K_MAX + 1):
            xCompactness = self.__computeKmeans(samples, k, mode, verbose)[0]
            xCompactness = self.__rangeCompactness(self.__reduceCompactness(xCompactness, baseCompactness))
            pCurve = (k, xCompactness)
            gap = prevCompactness - xCompactness
            self.__graph.add((k - 1, gap, prevCompactness))
            prevCompactness = xCompactness
            if gap < self.GAP_THRESHOLD and bestK == 0:
                bestK = k - 1
                if self.ALGO_OPTI:
                    break
        return bestK

    def __estimateKFK(self, samples, mode, verbose):
        """Estimate cluster count using f(K).
        """
        def fK(samples, thisk, skm1=0):
            nd = samples.shape[1]
            a = lambda k, nd: 1 - 3 / (4 * nd) if k == 2 else \
                a(k - 1, nd) + (1 - a(k - 1, nd)) / 6
            sk = self.__computeKmeans(samples, thisk, mode, verbose)[0]
            if thisk == 1:
                fs = 1
            elif skm1 == 0:
                fs = 1
            else:
                fs = sk / (a(thisk, nd) * skm1)
            return fs, sk

        fs = np.zeros(self.K_MAX - self.K_MIN)
        fs[0], sk = fK(samples, 1)
        self.__graph.add((1, fs[0]))
        for i, k in enumerate(xrange(self.K_MIN + 1, self.K_MAX)):
            fs[i + 1], sk = fK(samples, k, sk)
            self.__graph.add((k, fs[i + 1]))
        return (np.where(fs == min(fs))[0][0] + 1)

    def __wk(self, samples, centers, labels):
        """Sum of the pairwise distances for all points in clusters.
        """
        ulabels = np.unique(labels)
        k = len(ulabels)
        clusters = []
        for i in xrange(k):
            cluster = np.float64(samples[(labels.flatten() == ulabels[i])])
            clusters.append(cluster)
        return np.sum([(np.linalg.norm(centers[i] - c) ** 2) / (2.0 * c.shape[0]) \
                    for i in xrange(k) for c in clusters[i]])

    def __estimateKGapStat(self, samples, mode, verbose):
        """Estimate cluster count using gap statistic.
        """
        refCount = 5
        sampleCount, featureCount = samples.shape
        mins = samples.min(axis=0)
        maxs = samples.max(axis=0)
        dists = np.matrix(np.diag(maxs - mins))
        rands = np.float32(np.random.random_sample(size=(sampleCount, featureCount, refCount)))
        for i in range(refCount):
            rands[:,:,i] = rands[:,:,i] * dists + mins
        baseCompactness = None
        wkPrev, ewkPrev, skPrev = 0, 0, 0
        bestK = 0
        for (i, k) in enumerate(xrange(max(self.K_MIN, 2), self.K_MAX + 1)):
            compactness, labels, centers = self.__computeKmeans(samples, k, mode, verbose)
            baseCompactness = compactness if not baseCompactness else baseCompactness
            compactness = self.__reduceCompactness(compactness, baseCompactness)
            #compactness = self.__wk(samples, centers, labels)
            refCompactness = np.zeros((rands.shape[2],))
            for j in xrange(rands.shape[2]):
                refCompactness[j], labels, centers = self.__computeKmeans(rands[:,:,j], k, mode, verbose)
                #refCompactness[j] = self.__wk(rands[:,:,j], centers, labels)
                refCompactness[j] = self.__reduceCompactness(refCompactness[j], baseCompactness)
                #refCompactness[j] = np.log(refCompactness[j])
            wk = compactness #np.log(compactness)
            ewk = np.mean(refCompactness)
            sk = np.std(refCompactness) * np.sqrt(1 + 1 / refCount)
            delta = ((ewkPrev - wkPrev) - ((ewk - wk) - sk))
            if i >= 1:
                self.__graph.add((k - 1, wkPrev, ewkPrev, skPrev,
                                         (ewkPrev - wkPrev), delta))
                if delta > 0 and bestK == 0:
                    bestK = k - 1
                    if self.ALGO_OPTI:
                        self.__graph.add((k, wk, ewk, sk, 0, 0))
                        break
            wkPrev, ewkPrev, skPrev = wk, ewk, sk
        return bestK

    def __estimateKElbow(self, samples, mode, verbose):
        """Estimate cluster count using elbow.
        """
        firstCompactness = self.__computeKmeans(samples, self.K_MIN, mode, verbose)[0]
        baseCompactness = firstCompactness
        firstCompactness = self.__reduceCompactness(firstCompactness, baseCompactness)

        lastCompactness = self.__computeKmeans(samples, self.K_LAST_ELBOW, mode, verbose)[0]
        lastCompactness = self.__reduceCompactness(lastCompactness, baseCompactness)

        pFirst = np.float64(self.K_MIN), firstCompactness
        pLast = np.float64(self.K_LAST_ELBOW), lastCompactness

        m, p = self.__linearEquation(pFirst, pLast)
        self.__graph.addElbow(pFirst)
        bestK = (0, 0)

        found = False

        for k in xrange(self.K_MIN + 1, self.K_MAX):
            xCompactness = self.__computeKmeans(samples, k, mode, verbose)[0]
            xCompactness = self.__reduceCompactness(xCompactness, baseCompactness)
            pCurve = (np.float64(k), xCompactness)
            if pCurve == pFirst or pCurve == pLast:
                continue
            a = self.__distance(pCurve, pFirst)
            b = self.__distance(pFirst, pLast)
            c = self.__distance(pCurve, pLast)
            alpha = np.arccos(((a ** 2) + (b ** 2) - (c ** 2)) / (2.0 * a * b))
            d = np.sin(alpha) * a
            norm = np.sqrt(1.0 + (m ** 2))
            nvdir = (1.0 / norm, m / norm)
            dist = np.sqrt((a ** 2) - (d ** 2))
            h = (pFirst[0] + nvdir[0] * dist, pFirst[1] + nvdir[1] * dist)
            self.__graph.addElbow(pCurve, d=d, h=h)
            if (d >= bestK[1] and not found):
                bestK = int(pCurve[0]), d
            elif self.ALGO_OPTI:
                found = True
                break
        self.__graph.addElbow(pLast)
        return bestK[0]

    def __estimateK(self, samples, mode, kmode, verbose):
        """Return the best K for samples.
        """
        k = 1
        kmodes = { self.KMODE_ELBOW: self.__estimateKElbow,
                   self.KMODE_THRESHOLD: self.__estimateKThreshold,
                   self.KMODE_GAPSTAT: self.__estimateKGapStat,
                   self.KMODE_GAPTHRESHOLD: self.__estimateKGapThreshold,
                   self.KMODE_SLIDER: self.__estimateKSlider,
                   self.KMODE_FK: self.__estimateKFK }
        if kmode in kmodes:
            k = kmodes[kmode](samples, mode, verbose)

        k = 1 if k == 0 else k
        if verbose:
            print "Estimated cluster count: ", k
            print "Algorithm executed in {:.3f} s".format(time.time() - self.__startTime)
        return k

    def __kmeans(self, image, k, mode, kmode, features, verbose):
        """Apply K-Means algorithm on image.
        """
        samples = self.__getSamples(image, features)

        if kmode == self.KMODE_USER and k == 0:
            kmode == self.KMODE_GAP

        if kmode != self.KMODE_USER:
            k = self.__estimateK(samples, mode, kmode, verbose)

        retval, labels, centers = self.__computeKmeans(samples, k, mode, verbose)
        self.__graph.addData(image, samples, labels, centers, self.__orderedFeatures)

        return retval, labels, centers

    # Utils

    def __getSamples(self, image, flags):
        """Extract samples from features flags and stack them in an array.
        """
        samples = []
        for components, names, compute, table in self.__featuresTable:
            if self.__hasComponent(flags, components):
                samples.append((components, compute(image), table))

        del self.__orderedFeatures[:]
        features = None

        for i in xrange(flags.bit_length()):
            flag = ((flags >> i) & 1) << i
            if flag:
                for components, sample, table in samples:
                    if flag in components:
                        self.__orderedFeatures.append(flag)
                        features = table(flag, sample) if features is None else \
                                   np.hstack((features, table(flag, sample)))
        return features

    def __hasComponent(self, flags, components):
        """Return True if at least one item in components is in flags.
        """
        return sum([flags & f for f in components]) > 0
