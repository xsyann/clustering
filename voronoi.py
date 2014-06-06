#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
## voronoi.py
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Sat May 10 09:29:51 2014 xs_yann
## Last update Fri Jun  6 11:54:30 2014 xs_yann
##

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def circumCircle(pts):
    """Return the center of the circumcircle of the triangle defined by points.
    """
    rows, cols = pts.shape

    A = np.bmat([[2 * np.dot(pts, pts.T), np.ones((rows, 1))],
                 [np.ones((1, rows)), np.zeros((1, 1))]])

    b = np.hstack((np.sum(pts * pts, axis=1), np.ones((1))))
    x = np.linalg.solve(A,b)
    baryCoords = x[:-1]
    return np.sum(pts * np.tile(baryCoords.reshape((pts.shape[0], 1)), (1, pts.shape[1])), axis=0)

def voronoi(points):
    """Return line segments describing the voronoi diagram of points.
    """
    # Add four points at pseudo infinity
    m = np.max(np.abs(points)) * 1e5
    centers = np.vstack((points, np.array(((+m, +m), (-m, -m), (-m, +m), (+m, -m)))))

    delaunay = Delaunay(centers)
    triangles = delaunay.points[delaunay.vertices]
    circumCenters = [circumCircle(triangle) for triangle in triangles]

    segments = []
    for i, triangle in enumerate(triangles):
        circumCenter = circumCenters[i]
        for neighbor in delaunay.neighbors[i]:
            if neighbor != -1:
                segments.append((circumCenter, circumCenters[neighbor]))
    return segments
