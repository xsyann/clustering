## Clustering

Image clustering using Python OpenCV

![alt text](http://www.xsyann.com/epitech/clustering.png)

### Install

    git clone https://github.com/xsyann/clustering.git

### Usage

    python clustering.py
    

### Requirements
  
    cv2, numpy, matplotlib, scipy, urllib2, PyQt5
    
Matplotlib Qt5 compatibility: https://github.com/matplotlib/matplotlib/pull/3072

### Features

![alt text](http://www.xsyann.com/epitech/clustering/icon.png)

**Elbow**

- Draw the green line `l` passing through the first and the last point of the blue curve.
- Select the maximum length of the red segment perpendicular to the line `l` and passing through the curve point. 

![alt text](http://www.xsyann.com/epitech/clustering/elbow.png)

**Threshold**

- Reduce the compactness in range [0-1].
- Select the first point below the threshold defined in the Clusterer class.

![alt text](http://www.xsyann.com/epitech/clustering/threshold.png)

**Gap statistic**

- Gap statistic by Robert Tibshirani, Guenther Walther and Trevor Hastie.
- Stanford University, USA, November 2000.

![alt text](http://www.xsyann.com/epitech/clustering/gapstat.png)

**Gap Threshold**

- Reduce the compactness in range [0-1].
- Compute the gap between each neighbors point.
- Select the first point where the gap is below the threshold defined in the Clusterer class.

![alt text](http://www.xsyann.com/epitech/clustering/gapthreshold.png)

**f(k)**
- Selection of K in K-means clustering, D T Pham, S S Dimov, and C D Nguyen.
- Manufacturing Engineering Centre, Cardiff University, Cardiff, UK, 27 September 2004.

![alt text](http://www.xsyann.com/epitech/clustering/fk.png)

**Slider**

- Reduce the compactness and the cluster count in range [0-1].
- Draw a line between (0, 1) and (1, 0) and place a goal on this line depending of a percentage.
- Select the closest point to the goal.

![alt text](http://www.xsyann.com/epitech/clustering/slider.png)


**One clustering feature**

![alt text](http://www.xsyann.com/epitech/clustering/data1.png)

**Two clustering features**

![alt text](http://www.xsyann.com/epitech/clustering/data2.png)

**Three clustering features**

![alt text](http://www.xsyann.com/epitech/clustering/data3.png)


### References

http://blog.echen.me/2011/03/19/counting-clusters/
https://gist.github.com/michiexile/5635273
http://www.stanford.edu/~hastie/Papers/gap.pdf
http://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf
http://pami.uwaterloo.ca/groups/ras/PAMI_presentation.pdf
http://www.ima.umn.edu/~iwen/REU/paper4.pdf
http://docs.opencv.org/modules/core/doc/clustering.html
http://stackoverflow.com/questions/10650645/python-calculate-voronoi-tesselation-from-scipys-delaunay-triangulation-in-3d

