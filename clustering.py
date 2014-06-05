#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
## clustering.py
##
## Made by xs_yann
## Contact <contact@xsyann.com>
##
## Started on  Fri Apr 25 18:16:06 2014 xs_yann
## Last update Thu Jun  5 14:39:17 2014 xs_yann
##

import os
os.environ["QT_API"] = "pyqt5"

import sys
import cv2
import numpy as np
import clusterer
import urllib2
from clusterer import Clusterer
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton,
                             QSpinBox, QCheckBox, QHBoxLayout, QVBoxLayout,
                             QLabel, QLineEdit, QListWidget, QComboBox, QScrollArea,
                             QSplitter, QGroupBox, QTextEdit, QDesktopWidget,
                             QFrame, QColorDialog, QSizePolicy, QSlider)
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

class ClustererThread(QtCore.QThread):
    def __init__(self, mw):
        super(ClustererThread, self).__init__(mw)
        self.mw = mw

    def run(self):
        path = self.mw.sourcePathField.text()
        if not path:
            print "[Error] File path is empty"
            return
        try:
            img = Clusterer.readImage(path)
            imageBGRA = cv2.cvtColor(img, cv2.cv.CV_BGR2BGRA)
            self.mw.refreshSource(imageBGRA)
            features = self.mw.selectedFeatures
            if not features:
                return
            self.mw.clusterer = Clusterer()
            backgroundColor = self.mw.backgroundColor
            backgroundColor = backgroundColor.blue(), backgroundColor.green(), backgroundColor.red()
            if self.mw.transparentBg.isChecked():
                backgroundColor = None
            mode = self.mw.modeCombo.itemText(self.mw.modeCombo.currentIndex())
            mode = Clusterer.getModeByName(mode)
            modeK = self.mw.modeK.itemText(self.mw.modeK.currentIndex())
            modeK = Clusterer.getKModeByName(modeK)
            k = self.mw.clusterCount.value()
            self.mw.runButton.setEnabled(False)
            self.mw.clusters = self.mw.clusterer.getClusters(path, mode=mode,
                                                             kmode=modeK,
                                                             clusterCount=k,
                                                             features=features,
                                                             backgroundColor=backgroundColor,
                                                             slider=self.mw.clusterSlider.value())
            self.mw.currentCluster = 0
            self.mw.refreshCluster()
            self.mw.saveButton.setEnabled(True)

            self.mw.clusterer.graph(self.mw.figure)
            self.mw.canvas.setMinimumSize(self.mw.canvas.size())
            self.mw.canvas.draw()

        except (OSError, cv2.error, urllib2.HTTPError) as err:
            print err
        self.mw.runButton.setEnabled(True)

class Window(QWidget):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        # Set Features List
        self.selectedFeatures = Clusterer.getDefaultFeatures()
        self.availableFeatures = Clusterer.getAllFeatures()
        self.availableFeatures &= ~self.selectedFeatures

        self.initUI()

        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        self.clustererThread = ClustererThread(self)
        self.clusterer = None
        self.clusters = None
        self.currentCluster = 0
        self.backgroundColor = QtGui.QColor(0, 0, 0)

    def __del(self):
        sys.stdout = sys.__stdout__

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    ########################################################
    # Utils

    def np2Qt(self, imageRGBA):
        """Convert numpy array to QPixmap.
        """
        height, width, bytesPerComponent = imageRGBA.shape
        bytesPerLine = bytesPerComponent * width;

        qimg = QtGui.QImage(imageRGBA.data, width, height, bytesPerLine, QtGui.QImage.Format_ARGB32)
        return QtGui.QPixmap.fromImage(qimg)

    def checkerboard(self, size):
        """Create a checkboard.
        """
        h, w = size.height(), size.width()
        c0 = (191, 191, 191, 255)
        c1 = (255, 255, 255, 255)
        blocksize = 8
        coords = np.ogrid[0:h,0:w]
        idx = (coords[0] // blocksize + coords[1] // blocksize) % 2
        vals = np.array([c0, c1], dtype=np.uint8)
        return self.np2Qt(vals[idx])

    def fitImageToScreen(self, pixmap):
        """Fit pixmap to screen.
        """
        resolution = QDesktopWidget().screenGeometry()
        h, w = resolution.width(), resolution.height()
        w = min(pixmap.width(), w)
        h = min(pixmap.height(), h)
        return pixmap.scaled(QtCore.QSize(w, h), QtCore.Qt.KeepAspectRatio)

    def createFeaturesList(self, features):
        """Create feature list from features flags.
        """
        ql = []
        for i in xrange(features):
            flag = ((features >> i) & 1) << i
            if flag:
                ql.append(Clusterer.getFeatureName(flag))
        return ql

    def setPickerColor(self, color, colorPicker):
        """Set the color picker color.
        """
        css = 'QWidget { background-color: %s; border-width: 1px; \
        border-radius: 2px; border-color: black; border-style: outset; }'
        colorPicker.setStyleSheet(css % color.name())

    def refreshSource(self, img):
        """Display source image.
        """
        pixmap = self.np2Qt(img)
        pixmap = self.fitImageToScreen(pixmap)
        checkerboard = self.checkerboard(pixmap.size())

        painter = QtGui.QPainter()
        painter.begin(checkerboard)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        self.imageLabel.setPixmap(checkerboard)
        self.imageLabel.setFixedSize(pixmap.size())


    def refreshCluster(self):
        """Redraw image area with current cluster.
        """
        self.clusterCountLabel.setText(str(len(self.clusters)) + self.tr(' Clusters'))
        self.refreshSource(self.clusters[self.currentCluster])

    ########################################################
    # Signal handlers

    def normalOutputWritten(self, text):
        """Append text to debug infos.
        """
        cursor = self.debugText.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.debugText.setTextCursor(cursor)
        QApplication.processEvents()

    def run(self):
        """Run the clustering algorithm.
        """
        self.clustererThread.start()

    def save(self):
        """Save current clusters
        """
        path, filters = QFileDialog.getSaveFileName(self, self.tr('Open file'), '.',
                                                    self.tr('Image (*.jpg *.png *.jpeg)'))
        if path:
            for i, cluster in enumerate(self.clusters):
                fileName, fileExtension = os.path.splitext(path)
                clusterPath = fileName + "_{}".format(i) + fileExtension
                print "Saving file {}".format(clusterPath)
                self.np2Qt(cluster).save(clusterPath)


    def loadImage(self):
        """Open a file dialog.
        """
        path, filters = QFileDialog.getOpenFileName(self, self.tr('Open file'), '.',
                                                    self.tr('Image (*.jpg *.png *.jpeg)'))
        if path:
            self.sourcePathField.setText(path)

            pixmap = QPixmap(path)
            pixmap = self.fitImageToScreen(pixmap)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setFixedSize(pixmap.size())

    def removeFeature(self):
        """Remove selected features from selected features list.
        """
        selected = self.selectedFeatureList.selectedItems()
        for item in selected:
            feature = Clusterer.getFeatureByName(item.text())
            self.selectedFeatures &= ~feature
            self.availableFeatures |= feature
            self.selectedFeatureList.takeItem(self.selectedFeatureList.row(item))
            self.availableFeatureList.addItem(item)

    def addFeature(self):
        """Add selected features to selected features list.
        """
        selected = self.availableFeatureList.selectedItems()
        for item in selected:
            feature = Clusterer.getFeatureByName(item.text())
            self.availableFeatures &= ~feature
            self.selectedFeatures |= feature
            self.availableFeatureList.takeItem(self.availableFeatureList.row(item))
            self.selectedFeatureList.addItem(item)

    def colorDialog(self):
        """Open color dialog to pick a color.
        """
        color = QColorDialog.getColor()

        if color.isValid():
            self.backgroundColor = color
            self.setPickerColor(color, self.colorPicker)

    def toggleClusterCount(self, index):
        """Disable cluster count when 'auto' is checked.
        """
        mode = Clusterer.getKModeByName(self.modeK.itemText(self.modeK.currentIndex()))
        if mode == Clusterer.KMODE_USER:
            self.clusterCount.show()
        else:
            self.clusterCount.hide()

        if mode == Clusterer.KMODE_SLIDER:
            self.clusterSliderWidget.show()
        else:
            self.clusterSliderWidget.hide()

    def toggleDebugInfo(self, pressed):
        """Toggle debug infos widget.
        """
        if pressed:
            self.detailsScroll.show()
            self.showDetails.setText(self.tr('Details <<<'))
        else:
            self.detailsScroll.hide()
            self.showDetails.setText(self.tr('Details >>>'))

    def nextCluster(self):
        """Display the next cluster.
        """
        if self.clusters is None:
            return
        self.currentCluster += 1
        self.currentCluster %= len(self.clusters)
        self.refreshCluster()

    def prevCluster(self):
        """Display the previous cluster.
        """
        if self.clusters is None:
            return
        self.currentCluster -= 1
        self.currentCluster %= len(self.clusters)
        self.refreshCluster()

    def splitterMoved(self, pos, index):
        """Avoid segfault when QListWidget has focus and
        is going to be collapsed.
        """
        focusedWidget = QApplication.focusWidget()
        if focusedWidget:
            focusedWidget.clearFocus()

    def sliderMoved(self, value):
        """Set correct labels when slider moved.
        """
        self.clusterSliderLabel.setText('(' + str(100 - value) + '%)')
        self.compactnessSliderLabel.setText('(' + str(value) + '%)')

    ########################################################
    # Widgets

    def widgetSourcePath(self):
        """Create the widget for source path.
        """
        hbox = QHBoxLayout()

        sourcePathLabel = QLabel(self)
        sourcePathLabel.setText(self.tr('Source'))
        self.sourcePathField = QLineEdit(self)
        self.sourcePathField.setText('images/beach.jpg')
        sourcePathButton = QPushButton('...')
        sourcePathButton.clicked.connect(self.loadImage)
        hbox.addWidget(sourcePathLabel)
        hbox.addWidget(self.sourcePathField)
        hbox.addWidget(sourcePathButton)
        return hbox

    def widgetRun(self):
        """Create the run button.
        """
        hbox = QHBoxLayout()
        self.runButton = QPushButton(self.tr('Run'))
        self.runButton.clicked.connect(self.run)
        self.saveButton = QPushButton(self.tr('Save'))
        self.saveButton.clicked.connect(self.save)
        self.saveButton.setEnabled(False)
        hbox.addStretch(1)
        hbox.addWidget(self.saveButton)
        hbox.addWidget(self.runButton)
        hbox.addStretch(1)
        return hbox

    def widgetImage(self):
        """Create main image display.
        """
        imageArea = QHBoxLayout()
        scroll = QScrollArea()
        scroll.setAlignment(QtCore.Qt.AlignCenter)
        self.imageLabel = QLabel(self)

        scroll.setWidget(self.imageLabel)
        next = QPushButton(self.tr('>'))
        next.clicked.connect(self.nextCluster)
        prev = QPushButton(self.tr('<'))
        prev.clicked.connect(self.prevCluster)
        imageArea.addWidget(prev)
        imageArea.addWidget(scroll)
        imageArea.addWidget(next)

        vbox = QVBoxLayout()
        self.clusterCountLabel = QLabel(self)
        self.clusterCountLabel.setAlignment(QtCore.Qt.AlignCenter)
        f = QtGui.QFont('Arial', 14, QtGui.QFont.Bold);
        self.clusterCountLabel.setFont(f)
        vbox.addWidget(self.clusterCountLabel)
        vbox.addLayout(imageArea)
        return vbox

    def widgetFeatureList(self):
        """Create the features pair list widget.
        """
        hbox = QHBoxLayout()
        selectedFeatureList = QListWidget(self)
        selectedFeatureList.addItems(self.createFeaturesList(self.selectedFeatures))
        removeButton = QPushButton(self.tr('>>'))
        removeButton.clicked.connect(self.removeFeature)
        addButton = QPushButton(self.tr('<<'))
        addButton.clicked.connect(self.addFeature)
        availableFeatureList = QListWidget(self)
        availableFeatureList.addItems(self.createFeaturesList(self.availableFeatures))
        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addWidget(addButton)
        vbox.addWidget(removeButton)
        vbox.addStretch(1)
        vboxSelected = QVBoxLayout()
        selectedLabel = QLabel(self.tr('Selected'))
        selectedLabel.setAlignment(QtCore.Qt.AlignCenter)
        vboxSelected.addWidget(selectedLabel)
        vboxSelected.addWidget(selectedFeatureList)
        vboxAvailable = QVBoxLayout()
        availableLabel = QLabel(self.tr('Available'))
        availableLabel.setAlignment(QtCore.Qt.AlignCenter)
        vboxAvailable.addWidget(availableLabel)
        vboxAvailable.addWidget(availableFeatureList)

        hbox.addLayout(vboxSelected)
        hbox.addLayout(vbox)
        hbox.addLayout(vboxAvailable)

        self.selectedFeatureList = selectedFeatureList
        self.availableFeatureList = availableFeatureList

        return hbox

    def widgetParameters(self):
        """Create parameters widgets.
        """
        # Cluster Count
        self.autoK = QCheckBox(self.tr('Auto'))
        self.clusterCount = QSpinBox(self)
        self.clusterCount.setValue(2)
        self.clusterCount.setMinimum(1)
        self.modeK = QComboBox(self)

        hcluster = QHBoxLayout()
        hcluster.addWidget(QLabel(self.tr('Cluster count:')))
        hcluster.addWidget(self.modeK)
        hcluster.addWidget(self.clusterCount)

        # Slider
        hslider = QHBoxLayout()
        clusterLabel = QLabel(self.tr('Cluster count'))
        self.clusterSliderLabel = QLabel()
        compactnessLabel = QLabel(self.tr('Compactness'))
        self.compactnessSliderLabel = QLabel()
        self.clusterSlider = QSlider(QtCore.Qt.Horizontal)
        self.clusterSlider.valueChanged[int].connect(self.sliderMoved)
        self.clusterSlider.setMinimumWidth(100)
        self.clusterSlider.setValue(50)
        self.clusterSlider.setMaximum(100)
        hslider.addWidget(clusterLabel)
        hslider.addWidget(self.clusterSliderLabel)
        hslider.addWidget(self.clusterSlider)
        hslider.addWidget(compactnessLabel)
        hslider.addWidget(self.compactnessSliderLabel)
        self.clusterSliderWidget = QWidget()
        self.clusterSliderWidget.setLayout(hslider)

        # Set default mode
        self.modeK.currentIndexChanged.connect(self.toggleClusterCount)
        default = Clusterer.getDefaultKMode()
        defaultIndex = 0
        for i, (mode, name) in enumerate(Clusterer.getAllKModes()):
            if mode == default:
                defaultIndex = i
            self.modeK.addItem(name)
        self.modeK.setCurrentIndex(defaultIndex)

        # Algo
        combo = QComboBox(self)
        default = Clusterer.getDefaultMode()
        defaultIndex = 0
        for i, (mode, name) in enumerate(Clusterer.getAllModes()):
            if mode == default:
                defaultIndex = i
            combo.addItem(name)
        combo.setCurrentIndex(defaultIndex)
        halgo = QHBoxLayout()
        halgo.addWidget(QLabel(self.tr('Algorithm:')))
        halgo.addWidget(combo)
        self.modeCombo = combo

        # BG color
        color = QtGui.QColor(0, 0, 0)
        self.colorPicker = QPushButton('')
        self.colorPicker.setMaximumSize(QtCore.QSize(16, 16))
        self.colorPicker.clicked.connect(self.colorDialog)
        self.setPickerColor(color, self.colorPicker)
        self.transparentBg = QCheckBox(self.tr('Transparent'))
        self.transparentBg.setChecked(1)
        hbg = QHBoxLayout()
        hbg.addWidget(QLabel(self.tr('Background color:')))
        hbg.addWidget(self.colorPicker)
        hbg.addWidget(self.transparentBg)
        hbg.addStretch(1)

        # Features
        featureBox = QGroupBox(self.tr('Features'))
        features = self.widgetFeatureList()
        featureBox.setLayout(features)

        # Param Box
        paramBox = QGroupBox(self.tr('Parameters'))
        paramLayout = QVBoxLayout()
        paramLayout.addLayout(hcluster)
        paramLayout.addWidget(self.clusterSliderWidget)
        paramLayout.addLayout(halgo)
        paramLayout.addLayout(hbg)
        paramBox.setLayout(paramLayout)

        runButton = self.widgetRun()

        vbox = QVBoxLayout()
        vbox.addWidget(paramBox)
        vbox.addWidget(featureBox)
        vbox.addLayout(runButton)
        vbox.addStretch(1)

        return vbox

    def widgetDebug(self):
        """Create debug infos widget.
        """
        vbox = QVBoxLayout()
        self.debugText = QTextEdit(self)
        css = "QTextEdit { background-color: #FFF; color: #222 }"
        self.debugText.setStyleSheet(css)
        hbox = QHBoxLayout()
        self.showDetails = QPushButton(self.tr('Details >>>'))
        self.showDetails.setCheckable(True)
        self.showDetails.clicked[bool].connect(self.toggleDebugInfo)
        self.showDetails.setChecked(1)

        scroll = QScrollArea()
        scrollLayout = QVBoxLayout()
        scrollContents = QWidget()

        scroll.setWidgetResizable(True)
        scroll.setBackgroundRole(QtGui.QPalette.Dark);

        figureWidget = QWidget(scrollContents)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(figureWidget)
        self.toolbar = NavigationToolbar(self.canvas, self)

        plotLayout = QVBoxLayout()
        plotLayout.addWidget(self.canvas)
        plotLayout.addWidget(self.toolbar)
        figureWidget.setLayout(plotLayout)

        scrollLayout.addWidget(self.debugText)
        scrollLayout.addWidget(figureWidget)

        scrollContents.setLayout(scrollLayout)
        scroll.setWidget(scrollContents)

        self.detailsScroll = scrollContents

        self.canvas.setMinimumSize(self.canvas.size())
        self.toggleDebugInfo(True)

        hbox.addWidget(self.showDetails)
        hbox.addStretch(1)
        vbox.addLayout(hbox)
        vbox.addWidget(scroll)
        return vbox

    def initUI(self):
        """Create User Interface.
        """
        sourcePath = self.widgetSourcePath()
        parameters = self.widgetParameters()
        scroll = self.widgetImage()
        debug = self.widgetDebug()

        vbox1 = QVBoxLayout()
        vbox1.addLayout(sourcePath)
        hbox = QHBoxLayout()
        hbox.addLayout(parameters, 1)
        hbox.addLayout(scroll, 10)
        vbox1.addLayout(hbox)
        upSide = QWidget()
        upSide.setLayout(vbox1)

        vbox2 = QVBoxLayout()
        vbox2.addLayout(debug)
        downSide = QWidget()
        downSide.setLayout(vbox2)

        splitter = QSplitter(QtCore.Qt.Vertical)
        splitter.addWidget(upSide)
        splitter.addWidget(downSide)
        splitter.splitterMoved.connect(self.splitterMoved)

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(splitter)

        self.setLayout(mainLayout)

        self.setGeometry(300, 300, 300, 150)
        self.show()

def main():

    app = QApplication(sys.argv)

    main = Window()
    main.setWindowTitle('Clustering')
    main.setWindowIcon(QtGui.QIcon('icon.png'))
    main.show()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
