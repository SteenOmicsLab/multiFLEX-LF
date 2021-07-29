#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors: Pauline Hiort and Konstantin Kahnert
# Date: 2021_07_22
# Python version: 3.8

"""
multiFLEX-LF GUI version:
This script creates a graphical user interface for multiFLEX-LF using PyQT5
and connects it to the multiFLEX-LF_methods.py script
"""

##### import required libraries #####

##### import multiFLEX-LF methods
import multiFLEX_LF_methods as mFLF

##### libraries used to build the GUI
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton, QLabel, QLineEdit
from PyQt5.QtWidgets import QSpacerItem, QSpinBox, QDoubleSpinBox, QCheckBox, QSizePolicy
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QComboBox, QInputDialog
from PyQt5.QtCore import QThread, pyqtSignal, QSize, Qt, QMetaObject, QCoreApplication
from PyQt5.QtGui import QFont

##### used to exit multiFLEX-LF 
from sys import exit, argv

##### used for directory and file manipulation
from os import mkdir, remove
from os.path import isdir, exists, normpath
from subprocess import run, Popen

##### used for regular expressions
from re import findall

##### used for data manipulation
from pandas import read_csv, Series, DataFrame, MultiIndex
from numpy import nan, array, square, sqrt, warnings, isnan, ones, arange, flip, log10

##### used for parallel computation
from multiprocessing import Pool, freeze_support, cpu_count

##### used for runtime tracking
from time import time, sleep

##### used for calculation of statistics
from scipy.stats import f, median_abs_deviation
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster

##### used for ransac linear regression
from sklearn import linear_model
##### used for error handling
from sklearn.exceptions import UndefinedMetricWarning

##### used for plotting
from seaborn import histplot, scatterplot, lineplot, heatmap, color_palette
from matplotlib import use
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
from collections import Counter
from copy import copy
from plotly.io import renderers
from plotly.figure_factory import create_dendrogram
from plotly.graph_objects import Heatmap
##### use pdf output for matplotlib
use("pdf")
##### open plotly figures in default browser
renderers.default = "browser"


##### definition of classes #####

##### define worker thread class to execute the multiFLEX-LF method script in
class mFQLFWorkerThread(QThread):

    ##### initiate signals to communicate between main and worker thread
    sig_progress = pyqtSignal(float)
    sig_status = pyqtSignal(str)
    sig_run_button = pyqtSignal(bool)
    sig_cancel_button = pyqtSignal(bool)
    sig_error_reference = pyqtSignal(str)
    sig_error_file_open = pyqtSignal(str)
    sig_error_columns = pyqtSignal(str)
    sig_warning_plots = pyqtSignal(str)
    sig_warning_deseq = pyqtSignal(str)
    sig_input_dialog = pyqtSignal()
    
    ##### variable to save the user-defined clustering cutoff
    clust_cutoff = None

    ##### define init function
    def __init__(self, reference, input_file, input_file_name, path_output,
                 num_ransac_init, mod_cutoff, remove_outliers, create_plots, 
                 create_heatmaps, num_cpus, clustering, cosine_similarity, 
                 deseq_normalization, colormap, parent=None):
        QThread.__init__(self, parent)
        self.reference = reference
        self.input_file = input_file
        self.input_file_name = input_file_name
        self.path_output = path_output
        self.num_ransac_init = num_ransac_init
        self.mod_cutoff = mod_cutoff
        self.remove_outliers = remove_outliers
        self.create_plots = create_plots
        self.create_heatmaps = create_heatmaps
        self.num_cpus = num_cpus
        self.clustering = clustering
        self.cosine_similarity = cosine_similarity
        self.deseq_normalization = deseq_normalization
        self.colormap = colormap

    def send_progress(self, progress):
        """Sends progress as signal from worker thread to main thread"""
        self.sig_progress.emit(progress)
        
    def send_status(self, status):
        """Sends status as signal from worker thread to main thread"""
        self.sig_status.emit(status)

    def send_error_reference(self, name):
        """Sends reference sample error signal from worker thread to main thread"""
        self.sig_error_reference.emit(name)

    def send_error_file_open(self, path):
        """Sends file open error signal from worker thread to main thread"""
        self.sig_error_file_open.emit(path)

    def send_error_columns(self):
        """Sends group column signal from worker thread to main thread"""
        self.sig_error_columns.emit("")
    
    def send_warning_plots(self):
        """Sends plots warning signal from worker thread to main thread"""
        self.sig_warning_plots.emit("")
    
    def send_warning_deseq(self):
        """Sends deseq warning signal from worker thread to main thread"""
        self.sig_warning_deseq.emit("")

    def enable_run_button(self):
        """Sends signal from worker thread to main thread to make the run button clickable again"""
        self.sig_run_button.emit(True)

    def disable_cancel_button(self):
        """Sends signal from worker thread to main thread to make cancel button unclickable"""
        self.sig_cancel_button.emit(False)
        
    def open_input_dialog(self):
        """Send singnal to open input dialog and get the user input of the clustering cutoff
           returns the value given by the user"""
        self.sig_input_dialog.emit()
        
        ##### wait until user input is submitted and return the clustering cutoff
        while self.clust_cutoff == None:
            sleep(2)
        ##### save value to return
        cutoff = self.clust_cutoff
        ##### reset the variable for the next input prompt
        self.clust_cutoff = None
        
        return cutoff
    
    def stop(self):
        """Terminates the running process and resets progressbar as well as run and cancel buttons"""
        self.send_progress(0)
        self.send_status("Click 'Run' to Start multiFLEX-LF")
        self.enable_run_button()
        self.disable_cancel_button()
        self.terminate()

    def run(self):
        """Executes the multiFLEX-LF method script"""
        mFQLF_GUI_main(self, ui)



##### define main window class #####
class Ui_MainWindow(object):

    ##### define setup function
    def setupUi(self, MainWindow):
        """ Defines how the main window looks and creates it """
        ##### set object name
        MainWindow.setObjectName("MainWindow")

        ##### set fixed window size
        MainWindow.setMaximumSize(1000, 700)

        ##### initiate and name main window widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        ##### initiate and name GridLayout widget
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        
        ##### set font 
        bfont = QFont()
        bfont.setBold(True)
        bfont.setWeight(75)
        
        ##### initiate label for input file selection field
        self.label_input = QLabel(self.centralwidget)
        ##### set font 
        self.label_input.setFont(bfont)
        ##### define text type (rich text)
        self.label_input.setTextFormat(Qt.RichText)
        ##### name object
        self.label_input.setObjectName("label_input")
        ##### set grid position
        self.gridLayout.addWidget(self.label_input, 0, 0, 1, 4)
        ##### set hover text of the label
        self.label_input.setToolTip('Select path to input file in csv-format.')
        
        ##### initiate empty field to enter input file path
        self.lineEdit_input = QLineEdit(self.centralwidget)
        ##### set object name
        self.lineEdit_input.setObjectName("lineEdit_input")
        ##### set grid position
        self.gridLayout.addWidget(self.lineEdit_input, 2, 0, 1, 5)

        ##### initiate select input button
        self.pushButton_select_input = QPushButton(self.centralwidget)
        ##### set size of widget to a fixed size
        self.pushButton_select_input.setMinimumSize(QSize(120, 30))
        self.pushButton_select_input.setMaximumSize(QSize(120, 30))
        ##### set object name
        self.pushButton_select_input.setObjectName("pushButton_select_input")
        ##### connect to select_input_button_clicked function (sets what happens when clicked)
        self.pushButton_select_input.clicked.connect(self.select_input_button_clicked)
        ##### set grid position
        self.gridLayout.addWidget(self.pushButton_select_input, 2, 5, 1, 1)
        
        
        ##### initiate label for output folder path input field
        self.label_output = QLabel(self.centralwidget)
        ##### set font 
        self.label_output.setFont(bfont)
        ##### define text type (rich text)
        self.label_output.setTextFormat(Qt.RichText)
        ##### set object name
        self.label_output.setObjectName("label_output")
        ##### set grid position
        self.gridLayout.addWidget(self.label_output, 3, 0, 1, 4)
        ##### set hover text of the label
        self.label_output.setToolTip('Select folder/directory for the output files.')

        
        ##### initiate empty field to enter input file path
        self.lineEdit_output = QLineEdit(self.centralwidget)
        ##### set object name
        self.lineEdit_output.setObjectName("lineEdit_output")
        ##### set grid position
        self.gridLayout.addWidget(self.lineEdit_output, 4, 0, 1, 5)
        
        ##### initiate select output button
        self.pushButton_select_output = QPushButton(self.centralwidget)
        ##### set size of widget to a fixed size
        self.pushButton_select_output.setMinimumSize(QSize(120, 30))
        self.pushButton_select_output.setMaximumSize(QSize(120, 30))
        ##### set object name
        self.pushButton_select_output.setObjectName("pushButton_select_output")
        ##### connect to select_output_button_clicked function (sets what happens when clicked)
        self.pushButton_select_output.clicked.connect(self.select_output_button_clicked)
        ##### set grid position
        self.gridLayout.addWidget(self.pushButton_select_output, 4, 5, 1, 1)
        
        
        ##### initiate label for reference sample identifier input field
        self.label_reference = QLabel(self.centralwidget)
        ##### set font 
        self.label_reference.setFont(bfont)
        ##### define text type (rich text)
        self.label_reference.setTextFormat(Qt.RichText)
        ##### set object name
        self.label_reference.setObjectName("label_reference")
        ##### set grid position
        self.gridLayout.addWidget(self.label_reference, 5, 0, 1, 4)
        ##### set hover text of the label
        self.label_reference.setToolTip('Based on this, multiFLEX-LF determines which sample(s) to use as reference sample(s). This needs to match exactly the value in the "Group" column for all samples that shall be taken as reference. If more than one sample is given (e.g. in case of a control group) multiFLEX-LF calculates the median intensity over all reference samples for each peptide.<br>')

        
        ##### initiate empty field to enter reference sample identifier
        self.lineEdit_reference = QLineEdit(self.centralwidget)
        ##### set object name
        self.lineEdit_reference.setObjectName("lineEdit_reference")
        ##### set grid position
        self.gridLayout.addWidget(self.lineEdit_reference, 6, 0, 1, 5)


        ##### add spacer item between input field and options
        spacerItem_middle = QSpacerItem(1, 50, QSizePolicy.Fixed, QSizePolicy.Expanding)
        ##### set grid position
        self.gridLayout.addItem(spacerItem_middle, 7, 0, 1, 1)
        
        
        ##### initiate label for options section
        self.label_options = QLabel(self.centralwidget)
        ##### set font 
        self.label_options.setFont(bfont)
        ##### set text type
        self.label_options.setTextFormat(Qt.RichText)
        ##### set object name
        self.label_options.setObjectName("label_options")
        ##### set grid position
        self.gridLayout.addWidget(self.label_options, 8, 0, 1, 1)
        

        ##### initiate label for number of ransac initiations spinbox
        self.label_ransac_init = QLabel(self.centralwidget)
        ##### set object name
        self.label_ransac_init.setObjectName("label_ransac_init")
        ##### set grid position
        self.gridLayout.addWidget(self.label_ransac_init, 12, 0, 1, 1, Qt.AlignLeft)
        ##### set hover text of the label
        self.label_ransac_init.setToolTip('Number of times multiFLEX-LF/FLEXIQuant-LF fits a new RANSAC linear regression model to each sample to choose the best model. Between 5 and 100 initiations can be selected. <br>The more initiations, the higher the reproducibility and the probability that multiFLEX/FLEXIQuant-LF finds the optimal model. However, choosing a high number can significantly increases the runtime and more than 50 initiations rarely provide additional benefit.')

        
        ##### initiate a spinbox to select number of ransac initiations
        self.spinBox_ransac_init = QSpinBox(self.centralwidget)
        ##### set font size
        #self.spinBox_ransac_init.setStyleSheet("font: 8pt")
        ##### set size of widget to a fixed size
        self.spinBox_ransac_init.setMinimumSize(QSize(70, 30))
        self.spinBox_ransac_init.setMaximumSize(QSize(70, 30))
        ##### set minimum and maximum value that can be selected
        self.spinBox_ransac_init.setMinimum(5)
        self.spinBox_ransac_init.setMaximum(100)
        ##### set step size for clicking up or down
        self.spinBox_ransac_init.setSingleStep(5)
        ##### set default value to 30 initiations
        self.spinBox_ransac_init.setProperty("value", 30)
        ##### set object name
        self.spinBox_ransac_init.setObjectName("spinBox_ransac_init")
        ##### set grid position
        self.gridLayout.addWidget(self.spinBox_ransac_init, 12, 1, 1, 1)


        ##### initiate label for modification cutoff spinbox
        self.label_mod_cutoff = QLabel(self.centralwidget)
        ##### set object name
        self.label_mod_cutoff.setObjectName("label_mod_cutoff")
        ##### set grid position
        self.gridLayout.addWidget(self.label_mod_cutoff, 13, 0, 1, 1)
        ##### set hover text of the label
        self.label_mod_cutoff.setToolTip('RM score cutoff used to classify peptides as differentially modified. <br>')

        ##### initiate spinbox to select modification cutoff
        self.doubleSpinBox_mod_cutoff = QDoubleSpinBox(self.centralwidget)
        ##### set font size
        #self.doubleSpinBox_mod_cutoff.setStyleSheet("font: 8pt")
        ##### set size of widget to a fixed size
        self.doubleSpinBox_mod_cutoff.setMinimumSize(QSize(70, 30))
        self.doubleSpinBox_mod_cutoff.setMaximumSize(QSize(70, 30))
        ##### set number of decimals displayed
        self.doubleSpinBox_mod_cutoff.setDecimals(2)
        ##### set maximum value that can be selected
        self.doubleSpinBox_mod_cutoff.setMaximum(1.0)
        ##### set step size for clicking up and down
        self.doubleSpinBox_mod_cutoff.setSingleStep(0.05)
        ##### set default value to 0.5
        self.doubleSpinBox_mod_cutoff.setProperty("value", 0.5)
        ##### set object name
        self.doubleSpinBox_mod_cutoff.setObjectName("doubleSpinBox_mod_cutoff")
        ##### set grid position
        self.gridLayout.addWidget(self.doubleSpinBox_mod_cutoff, 13, 1, 1, 1)


        ##### set style sheet for check box
        StyleSheet_checkBox = '''
        QCheckBox::indicator {
            width:  20px;
            height: 20px;
        }
        '''
        
        ##### initiate label for checkbox for outlier peptides removal
        self.label_remove_outliers = QLabel(self.centralwidget)
        ##### set object name
        self.label_remove_outliers.setObjectName("label_remove_outliers")
        ##### set grid position
        self.gridLayout.addWidget(self.label_remove_outliers, 14, 0, 1, 1)
        ##### set hover text of the label
        self.label_remove_outliers.setToolTip('If checked, peptides with an outlier raw score above a computed cutoff are removed before RM score calculation. <br> If not checked, the outlier raw score is only removed for the specific sample, raw scores of the peptide of all other samples are kept for RM score analysis.')

        
        ##### initiate check box to select if outlier peptides should be removed or not
        self.checkBox_remove_outliers = QCheckBox(self.centralwidget)
        #####  use style sheet created above
        self.checkBox_remove_outliers.setStyleSheet(StyleSheet_checkBox)
        #####  set fixed size of widget
        self.checkBox_remove_outliers.setMinimumSize(QSize(25, 25))
        self.checkBox_remove_outliers.setMaximumSize(QSize(25, 25))
        #####  set orientation of text and checkbox
        self.checkBox_remove_outliers.setLayoutDirection(Qt.RightToLeft)
        #####  set text of check box to nothing
        self.checkBox_remove_outliers.setText("")
        #####  set default state to unchecked
        #self.checkBox_remove_outliers.setTristate(False)
        #self.checkBox_remove_outliers.setChecked(True)
        #####  set object name
        self.checkBox_remove_outliers.setObjectName("checkBox_remove_outliers")
        #####  set grid position
        self.gridLayout.addWidget(self.checkBox_remove_outliers, 14, 1, 1, 1, Qt.AlignRight)
        
        
        ##### initiate label for checkbox for plots generation
        self.label_create_plots = QLabel(self.centralwidget)
        ##### set object name
        self.label_create_plots.setObjectName("label_create_plots")
        ##### set grid position
        self.gridLayout.addWidget(self.label_create_plots, 15, 0, 1, 1)
        ##### set hover text of the label
        self.label_create_plots.setToolTip('If checked, a linear regression plot for each sample and a scatter plot is created for each protein. <br> All regression and scatter plots will be saved respectively in a single pdf file (_mFQ-LF-output_regression_plots.pdf and _mFQ-LF_scatter_plots.pdf). <br> <b>Warning:</b> This option only works with 1 CPU core!')

        
        ##### initiate check box to select if plots should be created or not
        self.checkBox_plots = QCheckBox(self.centralwidget)
        ##### use style sheet created above
        self.checkBox_plots.setStyleSheet(StyleSheet_checkBox)
        ##### set fixed size of widget
        self.checkBox_plots.setMinimumSize(QSize(25, 25))
        self.checkBox_plots.setMaximumSize(QSize(25, 25))
        ##### set orientation of text and checkbox
        self.checkBox_plots.setLayoutDirection(Qt.RightToLeft)
        ##### set text of check box to nothing
        self.checkBox_plots.setText("")
        ##### set default state to unchecked
        #self.checkBox_plots.setTristate(False)
        #self.checkBox_plots.setChecked(True)
        ##### set object name
        self.checkBox_plots.setObjectName("checkBox_plots")
        ##### set grid position
        self.gridLayout.addWidget(self.checkBox_plots, 15, 1, 1, 1, Qt.AlignRight)

        
        ##### initiate label for checkbox for the heatmaps generation
        self.label_create_heatmaps = QLabel(self.centralwidget)
        ##### set object name
        self.label_create_heatmaps.setObjectName("label_create_heatmaps")
        ##### set grid position
        self.gridLayout.addWidget(self.label_create_heatmaps, 16, 0, 1, 1)
        ##### set hover text of the label
        self.label_create_heatmaps.setToolTip('If checked, a heatmap of the RM scores of the peptides over the samples is created for each protein. <br> All plots will be saved in a single pdf file (_mFQ-LF_heatmaps.pdf).')

        
        ##### initiate check box to select if heatmaps should be created or not
        self.checkBox_heatmaps = QCheckBox(self.centralwidget)
        ##### use style sheet created above
        self.checkBox_heatmaps.setStyleSheet(StyleSheet_checkBox)
        ##### set fixed size of widget
        self.checkBox_heatmaps.setMinimumSize(QSize(25, 25))
        self.checkBox_heatmaps.setMaximumSize(QSize(25, 25))
        ##### set orientation of text and checkbox
        self.checkBox_heatmaps.setLayoutDirection(Qt.RightToLeft)
        ##### set text of check box to nothing
        self.checkBox_heatmaps.setText("")
        ##### set default state to unchecked
        #self.checkBox_heatmaps.setTristate(False)
        #self.checkBox_heatmaps.setChecked(True)
        ##### set object name
        self.checkBox_heatmaps.setObjectName("checkBox_heatmaps")
        ##### set grid position
        self.gridLayout.addWidget(self.checkBox_heatmaps, 16, 1, 1, 1, Qt.AlignRight)

        
        ##### initiate label for number of ransac initiations spinbox
        self.label_num_cpus = QLabel(self.centralwidget)
        ##### set object name
        self.label_num_cpus.setObjectName("label_num_cpus")
        ##### set grid position
        self.gridLayout.addWidget(self.label_num_cpus, 17, 0, 1, 1, Qt.AlignLeft)
        ##### set hover text of the label
        self.label_num_cpus.setToolTip('Number of CPUs/threads used for multiFLEX-LF computation. <br> Can be 2 times the number of available CPUs. <br> <b>Warning: This number should be lower than the maximum number of available CPUs, otherwise the computer might freeze!</b> ')

        
        ##### initiate a spinbox to select number of ransac initiations
        self.spinBox_num_cpus = QSpinBox(self.centralwidget)
        ##### set font size
        #self.spinBox_num_cpus.setStyleSheet("font: 8pt")
        ##### set size of widget to a fixed size
        self.spinBox_num_cpus.setMinimumSize(QSize(70, 30))
        self.spinBox_num_cpus.setMaximumSize(QSize(70, 30))
        ##### set minimum and maximum value that can be selected
        self.spinBox_num_cpus.setMinimum(1)
        self.spinBox_num_cpus.setMaximum(1000)
        ##### set step size for clicking up or down
        self.spinBox_num_cpus.setSingleStep(1)
        ##### set default value to 30 initiations
        self.spinBox_num_cpus.setProperty("value", 1)
        ##### set object name
        self.spinBox_num_cpus.setObjectName("spinBox_num_cpus")
        ##### set grid position
        self.gridLayout.addWidget(self.spinBox_num_cpus, 17, 1, 1, 1)
        
        
        ##### initiate a horizontal spacer item between FLEXIQuant-LF options and clustering options
        spacerItem_horizontal = QSpacerItem(200, 1, QSizePolicy.Expanding, QSizePolicy.Fixed)
        ##### set grid position
        self.gridLayout.addItem(spacerItem_horizontal, 8, 2, 1, 1)
        
        ##### initiate label for clustering options section
        self.label_clust_options = QLabel(self.centralwidget)
        ##### set font 
        self.label_clust_options.setFont(bfont)
        ##### set text type
        self.label_clust_options.setTextFormat(Qt.RichText)
        ##### set object name
        self.label_clust_options.setObjectName("label_clust_options")
        ##### set grid position
        self.gridLayout.addWidget(self.label_clust_options, 8, 3, 1, 1)        
        
        ##### initiate label for checkbox for clustering
        self.label_clustering = QLabel(self.centralwidget)
        ##### set object name
        self.label_clustering.setObjectName("label_clustering")
        ##### set grid position
        self.gridLayout.addWidget(self.label_clustering, 12, 3, 1, 1)
        ##### set hover text of the label
        self.label_clustering.setToolTip('If checked, computes a hierarchical clustering of the peptides based on their RM scores over the samples. <br> Results of the clustering are visualized in an interactive dendrogram and heatmap. This is saved in a html-file, which can be opened in any internet browser.')

        
        ##### initiate check box to select if clustering should be computed or not
        self.checkBox_clustering = QCheckBox(self.centralwidget)
        ##### use style sheet created above
        self.checkBox_clustering.setStyleSheet(StyleSheet_checkBox)
        ##### set fixed size of widget
        self.checkBox_clustering.setMinimumSize(QSize(25, 25))
        self.checkBox_clustering.setMaximumSize(QSize(25, 25))
        ##### set orientation of text and checkbox
        self.checkBox_clustering.setLayoutDirection(Qt.RightToLeft)
        ##### set text of check box to nothing
        self.checkBox_clustering.setText("")
        ##### set default state to unchecked
        self.checkBox_clustering.setTristate(False)
        self.checkBox_clustering.setChecked(True)
        ##### set object name
        self.checkBox_clustering.setObjectName("checkBox_clustering")
        ##### set grid position
        self.gridLayout.addWidget(self.checkBox_clustering, 12, 4, 1, 1, Qt.AlignRight)
        
        
        ##### initiate label for cosine similarity cutoff spinbox
        self.label_cos_similarity = QLabel(self.centralwidget)
        ##### set object name
        self.label_cos_similarity.setObjectName("label_cos_similarity")
        ##### set grid position
        self.gridLayout.addWidget(self.label_cos_similarity, 13, 3, 1, 1)
        ##### set hover text of the label
        self.label_cos_similarity.setToolTip('Cutoff score for the cosine similarity for missing RM scores imputation for the clustering. <br>Missing values of a peptide are imputed based on the median of the RM scores of the closest peptides to that peptide. The cosine similarity is computed to find all peptides with similarity above the given cosine similarity cutoff.')


        ##### initiate spinbox to select modification cutoff
        self.doubleSpinBox_cos_similarity = QDoubleSpinBox(self.centralwidget)
        ##### set font size
        #self.doubleSpinBox_cos_similarity.setStyleSheet("font: 8pt")
        ##### set size of widget to a fixed size
        self.doubleSpinBox_cos_similarity.setMinimumSize(QSize(70, 30))
        self.doubleSpinBox_cos_similarity.setMaximumSize(QSize(70, 30))
        ##### set number of decimals displayed
        self.doubleSpinBox_cos_similarity.setDecimals(2)
        ##### set maximum value that can be selected
        self.doubleSpinBox_cos_similarity.setMaximum(1.0)
        ##### set step size for clicking up and down
        self.doubleSpinBox_cos_similarity.setSingleStep(0.02)
        ##### set default value to 0.5
        self.doubleSpinBox_cos_similarity.setProperty("value", 0.98)
        ##### set object name
        self.doubleSpinBox_cos_similarity.setObjectName("doubleSpinBox_cos_similarity")
        ##### set grid position
        self.gridLayout.addWidget(self.doubleSpinBox_cos_similarity, 13, 4, 1, 1)
        

        
        ##### initiate label for checkbox for deseq2 normalization
        self.label_deseq_normalization = QLabel(self.centralwidget)
        ##### set object name
        self.label_deseq_normalization.setObjectName("label_deseq_normalization")
        ##### set grid position
        self.gridLayout.addWidget(self.label_deseq_normalization, 14, 3, 1, 1)
        ##### set hover text of the label
        self.label_deseq_normalization.setToolTip('If selected RM scores are normalized with DESeq2 before clustering. <br>Requires a working installation of R and DESeq2!')

        
        ##### initiate check box to select if deseq2 normalization should be computed or not
        self.checkBox_deseq_normalization = QCheckBox(self.centralwidget)
        ##### use style sheet created above
        self.checkBox_deseq_normalization.setStyleSheet(StyleSheet_checkBox)
        ##### set fixed size of widget
        self.checkBox_deseq_normalization.setMinimumSize(QSize(25, 25))
        self.checkBox_deseq_normalization.setMaximumSize(QSize(25, 25))
        ##### set orientation of text and checkbox
        self.checkBox_deseq_normalization.setLayoutDirection(Qt.RightToLeft)
        ##### set text of check box to nothing
        self.checkBox_deseq_normalization.setText("")
        ##### set default state to unchecked
        self.checkBox_deseq_normalization.setTristate(False)
        #self.checkBox_deseq_normalization.setChecked(True)
        ##### set object name
        self.checkBox_deseq_normalization.setObjectName("checkBox_deseq_normalization")
        ##### set grid position
        self.gridLayout.addWidget(self.checkBox_deseq_normalization, 14, 4, 1, 1, Qt.AlignRight)
        
        ##### initiate label for combo box of colormap for the heatmaps
        self.label_colormap = QLabel(self.centralwidget)
        ##### set object name
        self.label_colormap.setObjectName("label_colormap")
        ##### set grid position
        self.gridLayout.addWidget(self.label_colormap, 15, 3, 1, 1)
        ##### set hover text of the label
        self.label_colormap.setToolTip('Colormap used for the heatmap(s). Please choose one.')
        
        ##### intiante combo box for choosing the colormap for the heatmap
        self.box_colormap = QComboBox(self.centralwidget)
        ##### set size of widget to a fixed size
        self.box_colormap.setMinimumSize(QSize(150, 30))
        self.box_colormap.setMaximumSize(QSize(150, 30))
        ##### set orientation of text and checkbox
        #self.box_colormap.setLayoutDirection(Qt.RightToLeft)
        ##### define items for choosing
        self.box_colormap.addItem("red-white-blue")
        self.box_colormap.addItem("pink-white-green")
        self.box_colormap.addItem("purple-white-green")
        self.box_colormap.addItem("brown-white-bluegreen")
        self.box_colormap.addItem("orange-white-purple")
        self.box_colormap.addItem("red-white-grey")
        self.box_colormap.addItem("red-yellow-green")
        self.box_colormap.addItem("red-yellow-blue")
        self.box_colormap.setProperty("value", "red-white-blue")
        ##### set object name
        self.checkBox_deseq_normalization.setObjectName("box_colormap")
        
        ##### set grid position
        self.gridLayout.addWidget(self.box_colormap, 15, 4, 1, 2)
        
        
        ##### add spacer item between options and status bar
        spacerItem_middle2 = QSpacerItem(1, 50, QSizePolicy.Fixed, QSizePolicy.Expanding)
        ##### set grid position
        self.gridLayout.addItem(spacerItem_middle2, 19, 0, 1, 1)
        
        ##### initiate help button
        self.pushButton_help = QPushButton(self.centralwidget)
        ##### set size of widget to a fixed size
        self.pushButton_help.setMinimumSize(QSize(120, 30))
        self.pushButton_help.setMaximumSize(QSize(120, 30))
        ##### set object name
        self.pushButton_help.setObjectName("pushButton")
        ##### make button clickable
        self.pushButton_help.setEnabled(True)
        ##### connect to help_button_clicked function (sets what happens when clicked)
        self.pushButton_help.clicked.connect(self.help_button_clicked)
        ##### set grid position
        self.gridLayout.addWidget(self.pushButton_help, 26, 5, 1, 1)
        ##### set hover text of the label
        self.pushButton_help.setToolTip('Click to open the documentation.')

        
        ##### initiate status line
        self.statusBar = QtWidgets.QStatusBar()
        ##### set object name
        self.statusBar.setObjectName("statusBar")
        ##### set grid position
        self.gridLayout.addWidget(self.statusBar, 27, 0, 1, 4)
        ##### show message in status
        self.statusBar.showMessage("Click 'Run' to Start multiFLEX-LF")
        
        ##### initiate progress bar
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget, minimum=0, maximum=100)
        ##### set default value to 0
        self.progressBar.setProperty("value", 0)
        ##### set object name
        self.progressBar.setObjectName("progressBar")
        ##### set grid position
        self.gridLayout.addWidget(self.progressBar, 28, 0, 1, 4)
        
        ##### initiate run button
        self.pushButton_run = QPushButton(self.centralwidget)
        ##### set size of widget to a fixed size
        self.pushButton_run.setMinimumSize(QSize(120, 30))
        self.pushButton_run.setMaximumSize(QSize(120, 30))
        ##### set object name
        self.pushButton_run.setObjectName("pushButton_run")
        ##### make button clickable
        self.pushButton_run.setEnabled(False)
        ##### connect to run_button_clicked function (sets what happens when clicked)
        self.pushButton_run.clicked.connect(self.run_button_clicked)
        ##### set grid position
        self.gridLayout.addWidget(self.pushButton_run, 27, 4, 1, 1)

        ##### initiate cancel button
        self.pushButton_cancel = QPushButton(self.centralwidget)
        ##### set size of widget to a fixed size
        self.pushButton_cancel.setMinimumSize(QSize(120, 30))
        self.pushButton_cancel.setMaximumSize(QSize(120, 30))
        ##### set object name
        self.pushButton_cancel.setObjectName("pushButton_cancel")
        ##### make cancel button unclickable
        self.pushButton_cancel.setEnabled(False)
        ##### set grid position
        self.gridLayout.addWidget(self.pushButton_cancel, 27, 5, 1, 1)
        
        ##### initiate label for github repo link
        self.label_github_link = QLabel(self.centralwidget)
        ##### set font
        link_font = QFont()
        link_font.setPointSize(7)
        link_font.setBold(False)
        link_font.setItalic(False)
        link_font.setUnderline(True)
        link_font.setWeight(50)
        self.label_github_link.setFont(link_font)
        ##### set text type
        self.label_github_link.setTextFormat(Qt.RichText)
        ##### set maximum size, scaling and word wrap, not needed with fixed size window
        #self.label_github_link.setMaximumSize(QSize(16777215, 20))
        #self.label_github_link.setScaledContents(False)
        #self.label_github_link.setWordWrap(False)
        ##### set object name
        self.label_github_link.setObjectName("label_github_link")
        ##### add hyperlink to github repo
        self.label_github_link.setOpenExternalLinks(True)
        github_link = "<a href=\"https://gitlab.com/SteenOmicsLab/multiflex-lf\">GitLab Repository</a>"
        self.label_github_link.setText(github_link)
        ##### set grid position
        self.gridLayout.addWidget(self.label_github_link, 29, 5, 1, 1)
        
        ##### set central widget of mainwindow
        MainWindow.setCentralWidget(self.centralwidget)

        ##### call retranslateUi function
        self.retranslateUi(MainWindow)

        ##### connect slots by name
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        """
        Tries to translate and set the text that will be displayed in the GUI to the labels.
        If translation is not successful, english source text below will be displayed
        """
        
        _translate = QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "multiFLEX-LF"))
        
        self.pushButton_help.setText(_translate("MainWindow", "Help"))
        self.pushButton_cancel.setText(_translate("MainWindow", "Cancel"))
        self.pushButton_run.setText(_translate("MainWindow", "Run"))
        
        self.label_input.setText(_translate("MainWindow", "Input File"))
        self.pushButton_select_input.setText(_translate("MainWindow", "Select"))
        self.label_output.setText(_translate("MainWindow", "Output Folder"))
        self.pushButton_select_output.setText(_translate("MainWindow", "Select"))
        self.label_reference.setText(_translate("MainWindow", "Reference Sample Identifier"))
        
        self.label_options.setText(_translate("MainWindow", "FLEXIQuant-LF Options"))        
        self.label_ransac_init.setText(_translate("MainWindow", "RANSAC Initiations:"))        
        self.label_mod_cutoff.setText(_translate("MainWindow", "Modification Cutoff:"))
        self.label_remove_outliers.setText(_translate("MainWindow", "Remove Outlier Peptides:"))
        self.label_create_plots.setText(_translate("MainWindow", "Create Plots:"))
        self.label_create_heatmaps.setText(_translate("MainWindow", "Create Heatmaps:"))
        self.label_num_cpus.setText(_translate("MainWindow", "Number of CPUs:"))
        
        self.label_clust_options.setText(_translate("MainWindow", "Clustering Options"))
        self.label_clustering.setText(_translate("MainWindow", "Hierarchical Clustering:"))
        self.label_cos_similarity.setText(_translate("MainWindow", "Cosine Similarity Cutoff:"))
        self.label_deseq_normalization.setText(_translate("MainWindow", "DESeq2 Normalization:"))
        self.label_colormap.setText(_translate("MainWindow", "Colormap for Heatmap:"))
        

    def select_input_button_clicked(self):
        """
        Sets what happens when the select input button is clicked
        Opens a file browser where the user can select the input file (only displays .csv files)
        displays the file path of the selected file in input path field and the folder path in output folder field
        checks if input file and output folder filed are filled and paths are valid
        and if true, makes run button clickable
        """
        
        ##### initiate QFileDialog widget
        filedialog = QFileDialog()

        ##### open file browser and save path of selected file
        self.fpath_input = filedialog.getOpenFileName(filedialog, filter="CSV files (*.csv)")[0]

        ##### display path of select file in lineEdit_input
        self.lineEdit_input.setText(self.fpath_input)

        ##### get file name
        self.fname_input = findall("[^\\\\,/]*$", self.fpath_input)[0]

        ##### get output folder
        self.folder_output = self.fpath_input[0:-len(self.fname_input)-1]

        ##### display input folder in lineEdit_output (as default)
        self.lineEdit_output.setText(self.folder_output)

        ##### check if input and output fields are filled
        if exists(self.folder_output) is True and exists(self.fpath_input) is True:
            self.pushButton_run.setEnabled(True)


    def select_output_button_clicked(self):
        """ 
        Sets what happens when the select output button is clicked
        Opens a file browser where the user can select the output folder
        displays the file path of the selected file in input path field and 
        the folder path in output folder field
        checks if input file and output folder filed are filled and paths are valid
        and if true, makes run button clickable
        """
        
        ##### initiate QFileDialog widget
        filedialog = QFileDialog()

        ##### open file browser and save path of selected file
        self.folder_output = filedialog.getExistingDirectory(filedialog, "Select Directory")

        ##### display path of select folder in lineEdit_output
        self.lineEdit_output.setText(self.folder_output)

        ##### check if input and output fields are filled
        if exists(self.folder_output) is True and exists(self.fpath_input) is True:
            self.pushButton_run.setEnabled(True)
    
    
    def error_columns(self, key):
        """ Displays error message and resets progressbar and run and cancel buttons """
        
        ##### display error message
        self.display_error_msg("Incorrect input format!\n" + "At least one of the columns \"ProteinID\", \"PeptideID\", \"Sample\", \"Group\" or \"Intensity\" could not be found in the input file.")
        ##### reset progressbar and run and cancel buttons
        self.reset_after_run_error()


    def error_reference(self, name):
        """ Displays error message and resets progressbar and run and cancel buttons """
        
        ##### display error message
        self.display_error_msg("Given reference sample identifier \"" + name + "\" not found in \"Group\" column")
        ##### reset progressbar and run and cancel buttons
        self.reset_after_run_error()


    def error_file_open(self, path):
        """ Displays error message and resets progressbar and run and cancel buttons """
        
        ##### display error message
        self.display_error_msg("Permission denied!\n" + "Please close " + path)
        ##### reset progressbar and run and cancel buttons
        self.reset_after_run_error()

    def warning_plots(self, key):
        """ Displays warning message """
        
        ##### display error message
        self.display_warning_msg("Can not create regression and scatter plots with multiprocessing!\nPlease use -cpu 1 to create regression and scatter plots!\nContinuing without creating the plots.")

    def warning_deseq(self, key):
        """ Displays warning message """
        
        ##### display error message
        self.display_warning_msg("DESeq2 Normalization did not work. Continuing without!")

    
    def reset_after_run_error(self):
        """ Resets progressbar and run and cancel buttons """
        
        ##### enable start button
        self.update_run_button(True)
        ##### disable cancel button
        self.pushButton_cancel.setEnabled(False)
        ##### set progress bar to 0
        self.progressBar.setValue(0)
        ##### set status to initial message
        self.statusBar.showMessage("Click 'Run' to Start multiFLEX-LF")


    def update_progress(self, progress):
        """
        Updates the progressbar to the given integer (progress)
        and displays message when program has finished
        """
        
        ##### while clustering set max of progress to 0 
        if progress == -1:
            self.progressBar.setMaximum(0)
        elif progress == 0:
            self.progressBar.setMaximum(100)
        
        ##### set progress bar to given value
        self.progressBar.setValue(int(progress))
        
        ##### if done
        if progress == 100:
            self.progressBar.setMaximum(100)
            self.progressBar.setValue(int(progress))
            self.display_done_msg()


    def update_status(self, status):
        """
        Updates the progressbar to the given integer (progress)
        and displays message when program has finished
        """
        
        self.statusBar.showMessage(str(status))


    def update_run_button(self, on_off):
        """
        Makes run button clickable or unclickable
        Arguments:
        True: clickable
        False: unclickable
        """
        
        self.pushButton_run.setEnabled(on_off)


    def update_cancel_button(self, on_off):
        """
        Makes cancel button clickable or unclickable
        Arguments:
        True: clickable
        False: unclickable
        """
        
        self.pushButton_cancel.setEnabled(on_off)


    def display_done_msg(self):
        """
        Opens dialog window with task completion message and asks user to 
        open output folder
        """
        
        ##### enable start button
        self.update_run_button(True)

        ##### disable cancel button
        self.pushButton_cancel.setEnabled(False)

        ##### initiate dialog box
        self.msg = QMessageBox()
        ##### set message box type
        self.msg.setIcon(QMessageBox.Information)
        ##### set window title
        self.msg.setWindowTitle("Task completed")
        ##### set text
        self.msg.setText("Done!")
        ##### set informative text
        self.msg.setInformativeText("Open output folder?")
        ##### add yes and no buttons
        self.msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        ##### set yes button as default selection
        self.msg.setDefaultButton(QMessageBox.Yes)

        ##### display dialog
        reply = self.msg.exec_()

        ##### if Yes is clicked, open output folder
        if reply == QMessageBox.Yes:
            ##### open output folder
            output_dir = normpath(self.path_output)
            Popen(r'explorer "' + output_dir + '"')
        else:
            pass


    def display_error_msg(self, msg_txt):
        """
        Opens dialog window and displays error message with msg_text
        msg_text: str, error message to be displayed
        """
        
        ##### initiate message box
        self.msg = QMessageBox()
        ##### set icon
        self.msg.setIcon(QMessageBox.Critical)
        ##### set text
        self.msg.setText("Error")
        ##### set informative text
        self.msg.setInformativeText(msg_txt)
        ##### set window title
        self.msg.setWindowTitle("Error")
        ##### display error message window
        self.msg.exec_()
        
        
    def display_warning_msg(self, msg_txt):
        """
        Opens dialog window and displays warning message with msg_text
        msg_text: str, warning message to be displayed 
        """
        
        ##### initiate message box
        self.msg = QMessageBox()
        ##### set icon
        self.msg.setIcon(QMessageBox.Warning)
        ##### set text
        self.msg.setText("Warning")
        ##### set informative text
        self.msg.setInformativeText(msg_txt)
        ##### set window title
        self.msg.setWindowTitle("Warning")
        ##### display error message window
        self.msg.exec_()


    def help_button_clicked(self):
        """ Opens documentation """
        Popen([r'multiFLEX-LF_Documentation.pdf'], shell=True)
    
    
    def input_dialog_opened(self):
        """ Opens an input dialog and asks for a clustering distance cutoff 
            which is applied to build clusters from the hierarchical clustering"""
        ##### start input dialog
        self.input_dialog = QInputDialog()
        
        ##### open input dialog and get the users input upon clicking the buttons
        self.value, self.okClicked = self.input_dialog.getDouble(None, "multiFLEX-LF Clustering Distance Cutoff","Enter a clustering distance cutoff. \n\nYou can determine a cutoff for your data from the dendrogram and heatmap. \nThe figure can be found and investigated in the file with the \nsuffix mFQ-LF_RM_scores_clustered_heatmap.html \nPlease use the positive value corresponidng to the negative distance shown in the dendrogram! \n\nThe distance will be used to build the flat clusters and assign a corresponding \nCluster ID to each peptide. \n\nThis window will open again, if you want to use another cutoff. \nPress Cancel if you do not want to apply a clustering distance cutoff.", 0, 0, 100, 2)
        
        ##### if OK clicked change the variable clust_cutoff to the input value
        ##### otherwise change it to 'q'
        if self.okClicked:
            self.worker_thread.clust_cutoff = self.value
        elif not self.okClicked:
            self.worker_thread.clust_cutoff = 'q'

    
    def run_button_clicked(self):
        """
        Sets what happens when the run button is clicked
        collects all paths and parameters set by the user and displays error message 
        if paths don't exist starts the multiFLEX_LF_method script in a new thread 
        with the given parameters
        """
        ##### get reference samples
        reference = self.lineEdit_reference.text()

        ##### get path of input file
        input_file = self.lineEdit_input.text()

        ##### get file name
        input_file_name = self.fname_input

        ##### get path of output folder
        self.path_output = self.lineEdit_output.text()

        ##### get number of ransac initiations
        num_ransac_init = self.spinBox_ransac_init.value()

        ##### get modification cutoff
        mod_cutoff = self.doubleSpinBox_mod_cutoff.value()
        
        ##### get checkBox_remove_outliers value
        if self.checkBox_remove_outliers.isChecked() == True:
            remove_outliers = True
        else:
            remove_outliers = False

        ##### get checkBox_plots value
        if self.checkBox_plots.isChecked() == True:
            create_plots = True
        else:
            create_plots = False
        
        ##### get checkBox_heatmaps value
        if self.checkBox_heatmaps.isChecked() == True:
            create_heatmaps = True
        else:
            create_heatmaps = False
            
        num_cpus = self.spinBox_num_cpus.value()
            
        ##### get checkBox_clustering value
        if self.checkBox_clustering.isChecked() == True:
            clustering = True
        else:
            clustering = False
            
        ##### get cosine similarity cutoff
        cosine_similarity = self.doubleSpinBox_cos_similarity.value()
            
        ##### get checkBox_deseq_normalization value
        if self.checkBox_deseq_normalization.isChecked() == True:
            deseq_normalization = True
        else:
            deseq_normalization = False
            
        ##### get colormap value
        colormap = self.box_colormap.currentText()
        
        ##### check if input and output fields are filled
        if exists(input_file) is True and exists(self.path_output) is True:

            ##### disable start button
            self.update_run_button(False)

            ##### enable cancel button
            self.pushButton_cancel.setEnabled(True)

            ##### initiate worker thread with parameters set by user
            self.worker_thread = mFQLFWorkerThread(reference, input_file, input_file_name, 
                                                  self.path_output, num_ransac_init, mod_cutoff, 
                                                  remove_outliers, create_plots, create_heatmaps, 
                                                  num_cpus, clustering, cosine_similarity, 
                                                  deseq_normalization, colormap)
            ##### start worker thread
            self.worker_thread.start()

            ##### connect cancel button to terminate thread function
            self.pushButton_cancel.clicked.connect(self.worker_thread.stop)

            ##### connect to all signals from worker_thread
            self.worker_thread.sig_cancel_button.connect(self.update_cancel_button)
            self.worker_thread.sig_run_button.connect(self.update_run_button)
            self.worker_thread.sig_error_reference.connect(self.error_reference)
            self.worker_thread.sig_error_file_open.connect(self.error_file_open)
            self.worker_thread.sig_error_columns.connect(self.error_columns)
            self.worker_thread.sig_warning_plots.connect(self.warning_plots)
            self.worker_thread.sig_warning_deseq.connect(self.warning_deseq)
            self.worker_thread.sig_progress.connect(self.update_progress)
            self.worker_thread.sig_status.connect(self.update_status)
            self.worker_thread.sig_input_dialog.connect(self.input_dialog_opened)

        ##### if output folder does not exist, display error message
        elif exists(input_file) is True and exists(self.path_output) is False:
            self.display_error_msg("Specified output folder does not exist!")

        ##### if input file does not exist, display error message
        elif exists(input_file) is False and exists(self.path_output) is True:
            self.display_error_msg("Specified input file does not exist!")

        ##### if input file and output folder do not exist, display error message
        else:
            self.display_error_msg("Specified input file and output folder do not exist!")



        

def mFQLF_GUI_main(self, ui):
    """
    multiFLEX-LF GUI version
    
    For each protein in a given dataset, the FLEXIQuant-LF method is apllied:
    - multiple robust linear regression models (using RANSAC) are fitted for each sample 
        with median reference sample peptide intensities as independent variable and 
        sample peptide intensities as dependent variable and selects the best model
    - vertical distances to the regression line are calculated for each peptide datapoint
    - raw scores are calculated by dividing the distances by the median reference sample 
        intensity times the slope of the regression line and subtracting the results from 1
    - optionally: for each sample, raw scores of peptides are removed resulting in raw scores 
        higher than the median of all raw scores of a sample plus three times the 
        median absolute deviation of all raw scores of a sample
    - RM score are calculated by applying TOP3 normalization to the filtered raw scores
    For all proteins together this script:
    - optional: hierarchical clustering of all peptides based on their RM scores
    - missing RM scores are imputed by the median of the RM scores of the closest peptides
        based in the cosine similarity (cutoff)
    - optional: DESeq2 normalization of the RM scores can be calculated (requires working installation of R)
    - input prompt for a clustering distance cutoff is given at the end, based on the cutoff
        the clusters from the hierarchical clustering are created
    
    Output:
        - mFQ-LF-output_raw_scores.csv: 
            csv file of the raw scores for all proteins and peptides
            plus Slope, R2 model (based on RANSAC inliers), R2 data (based on all data points), 
            Reproducibility factor (fraction of the most frequently resulting slope of all RANSAC initiations)
        - mFQ-LF-output_RM_scores.csv: 
            csv file of the RM scores for the proteins and the peptides 
        - mFQ-LF-output_diff_modified.csv: 
            boolean matrix of same dimention as output_RM_scores.
            True: peptide is differentially modified (RM score smaller than modification cutoff)
            False: peptide is not differentially modified (RM score equal or larger than modification cutoff)
        - mFQ-LF-output_removed_peptides.csv: 
            contains all peptides that were removed during raw score filtering
        - mFQ-LF_regression_plots.pdf and mFQ-LF_scatter_plots.pdf: 
            multi-page pdf files contain the regression plots and scatter plots (log of intensity vs. RM score)
            each page shows the plot of one protein
        - mFQ-LF_RM_scores_distribution.pdf:
            pdf file of the RM score distribution of the samples for each group
        - mFQ-LF_heatmaps.pdf:
            pdf file of a heatmap of the RM scores of each protein
            - mFQ-LF-output_RM_scores_DESeq.csv:
            csv file of the DESeq2 normalized RM scores
        - mFQ-LF_RM_scores(_DESeq)_clustered_heatmap.html:
            html file of the interactive dendrogram and heatmap of the hierachical clustering
        - mFQ-LF_RM_scores(_DESeq)_clustered.csv:
            csv file of the RM scores of all proteins and peptides sorted in the 
            hierarchical clustering order with the ID shown in the interactive figure
        - mFQ-LF_RM_scores(_DESeq)_clustered_heatmap_dist-x.html:
            html file of the interactive dendrogram and heatmap of the hierachical clustering
            with the cluster of the user given distance (x) color coded in the dendrogram
        - mFQ-LF_RM_scores(_DESeq)_clustered_dist-x.csv:
            csv file of the RM scores of all proteins and peptides sorted in the 
            hierarchical clustering order with the ID shown in the interactive figure
            with the cluster ids of the user given distance (x)
        
    Parameters:
        This function is imported to the multiFLEX_LF_GUI script and run within the FQLFWorkerThread class
        self: FQLFWorkerThread class object from where the function is executed
        ui: Object of class Ui_MainWindow
    """
    #### get current time to track the runtime
    starttime = time()
    
    ##### load data in pandas dataframes
    df_input = read_csv(self.input_file, sep=",", index_col=None)
    
    ##### check if ProteinID, PeptideID, Group, Sample, Intensity columns exists
    try:
        df_input["ProteinID"]
        df_input["PeptideID"]
        df_input["Group"]
        df_input["Sample"]
        df_input["Intensity"]
    except KeyError:
        ##### print error message
        self.send_error_columns()
        ##### enable button
        self.enable_run_button()
        ##### terminate process
        exit(ui.worker_thread.exec_())
        

    ##### check given reference identifier exists in group column
    if str(self.reference) not in set(df_input["Group"].astype(str)):
        ##### print error message
        self.send_error_reference(self.reference)
        ##### enable button
        self.enable_run_button()
        ##### terminate process
        exit(ui.worker_thread.exec_())
        
    ##### create intensity matrix from imput table
    ##### ProteinID and PeptidesID as column indices; Group and Sample as row indices
    df_intens_matrix_all_proteins = df_input.set_index(["ProteinID", "PeptideID", "Group", "Sample"]).unstack(level=["Group", "Sample"]).T
    df_intens_matrix_all_proteins = df_intens_matrix_all_proteins.set_index([df_intens_matrix_all_proteins.index.get_level_values("Group"), df_intens_matrix_all_proteins.index.get_level_values("Sample")])
    df_intens_matrix_all_proteins = df_intens_matrix_all_proteins.sort_index(axis=0).sort_index(axis=1)
    
    ##### send status to statusbar
    self.send_status("Data import finished")

    ##### check if protein-wise regression and scatter plots should and can be created
    ##### Construction of the regression and scatter plots does not work with multiprocessing!
    if self.create_plots and self.num_cpus < 2:
        
        ##### create output file path for the pdf file of the regression plots
        path_regression_plots = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_regression_plots.pdf")
        
        ##### try to access the pdf and raise permission error if not possible
        try: reg_plots_pdf = PdfPages(path_regression_plots)
        except PermissionError:
            ##### print error message
            self.send_error_file_open(path_regression_plots)
            ##### terminate process
            exit(ui.worker_thread.exec_())
        
        ##### create title page for regression plots
        fig = plt.figure(figsize=(1, 1))
        plt.title("RANSAC Linear Regression Plots of multiFLEX-LF/FLEXIQuant-LF", fontsize=20)
        plt.axis("off")
        reg_plots_pdf.savefig(figure=fig, bbox_inches="tight", dpi=300)
        plt.close()
        
        ##### create output file path for the pdf file of the intensities vs. RM scores scatter plot
        path_scatter_plots = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_scatter_plots.pdf")
        
        ##### try to access the pdf file and raise permission error if not possible
        try: scatter_plots_pdf = PdfPages(path_scatter_plots)
        except PermissionError:
            ##### print error message
            self.send_error_file_open(path_scatter_plots)
            ##### terminate process
            exit(ui.worker_thread.exec_())
        
        ##### create title page for scatter plots
        fig = plt.figure(figsize=(1, 1))
        plt.title("Peptide Intensity vs. RM score Scatter Plots of multiFLEX-LF", fontsize=20)
        plt.axis("off")
        scatter_plots_pdf.savefig(figure=fig, bbox_inches="tight", dpi=300)
        plt.close()
        
    else:
        
        ##### if multiprocessing is activated show warning
        if self.create_plots:
            self.send_warning_plots()
            
        ##### creating the plots does not work with multiprocessing!
        ##### turn it off and continue without plot creation
        self.create_plots = False
        reg_plots_pdf = ''
        scatter_plots_pdf = ''
    
    
    ##### create output file path for the pdf file of the RM score distribution
    path_distribution_plots = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_distribution.pdf")
    ##### try to access the pdf file and raise permission error if not possible
    try: distri_plots_pdf = PdfPages(path_distribution_plots)
    except PermissionError:
        ##### print error message
        self.send_error_file_open(path_distribution_plots)
        ##### terminate process
        exit(ui.worker_thread.exec_())
    
    if self.deseq_normalization:
        ##### create output file path for the pdf file of the RM score distribution after DESeq2 normalization
        path_deseq_distribution_plots = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_DESeq_distribution.pdf")
        ##### try to access the pdf file and raise permission error if not possible
        try: deseq_distri_plots_pdf = PdfPages(path_deseq_distribution_plots)
        except PermissionError:
            ##### print error message
            self.send_error_file_open(path_deseq_distribution_plots)
            ##### terminate process
            exit(ui.worker_thread.exec_())
        
    
    ##### check if heatmaps should and can be created
    if self.create_heatmaps:
        path_heatmaps = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_heatmaps.pdf")
        ##### try to access the pdf file and raise permission error if not possible
        try: heatmaps_pdf = PdfPages(path_heatmaps)
        except PermissionError:
            ##### print error message
            self.send_error_file_open(path_heatmaps)
            ##### terminate process
            exit(ui.worker_thread.exec_())
    
    ##### create header for csv output files
    header = df_intens_matrix_all_proteins.T.droplevel("Group", axis=1).head(0)
    
    ##### save raw scores as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_raw_scores.csv")
    try: header.to_csv(path_out, sep=',', mode='w', index=True, header=True, float_format="%.5f")
    except PermissionError:
        ##### print error message
        self.send_error_file_open(path_out)
        ##### terminate process
        exit(ui.worker_thread.exec_())

    ##### save RM scores as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_RM_scores.csv")
    try: header.to_csv(path_out, sep=',', mode='w', index=True, header=True, float_format="%.5f")
    except PermissionError:
        ##### print error message
        self.send_error_file_open(path_out)
        ##### terminate process
        exit(ui.worker_thread.exec_())

    ##### save differentially modified dataframe as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_diff_modified.csv")
    try: header.to_csv(path_out, sep=',', mode='w', index=True, header=True)
    except PermissionError:
        ##### print error message
        self.send_error_file_open(path_out)
        ##### terminate process
        exit(ui.worker_thread.exec_())
    
    ##### create header for output file containing removed peptides
    removed_header = DataFrame([{"ProteinID": 0, "PeptideID": 0}])
    
    ##### save removed peptides as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_removed_peptides.csv")
    try: removed_header.head(0).to_csv(path_out, sep=',', mode='w', index=False, header=True)
    except PermissionError:
        ##### print error message
        self.send_error_file_open(path_out)
        ##### terminate process
        exit(ui.worker_thread.exec_())

    ##### remove the header variables
    del header, removed_header
    
    
    ##### start multiFLEX-LF/FLEXIQuant-LF computation #####
    
    ##### dataframe for the calculated RM scores for all proteins and peptides
    df_RM_scores_all_proteins = DataFrame()
    
    ##### create a list of all proteins in the data set
    list_proteins = df_intens_matrix_all_proteins.columns.get_level_values("ProteinID").unique().sort_values()  
    
    ##### determine progress step size and set progress to 0
    num_proteins = len(list_proteins)
    progress_step = (99/num_proteins)
    progress = 0
    self.send_progress(progress)
    
    ##### if number of cpus smaller than 2 continue with unparallel processing
    ##### otherwise use parallel processing to analyze all proteins
    if self.num_cpus < 2:
        
        ##### send status to statusbar
        self.send_status("FLEXIQuant-LF computation...")
        
        for protein in list_proteins:
            
            ##### run the FLEXIQuant-LF method (run_FQLF) with the current protein
            df_RM_scores = mFLF.run_FQLF(df_intens_matrix_all_proteins[protein].copy(), protein, str(self.reference), 
                             self.num_ransac_init, self.mod_cutoff, self.remove_outliers, self.input_file_name, 
                             self.path_output, reg_plots_pdf, scatter_plots_pdf)
            
            ##### add RM scores of protein to dataframe of all proteins
            df_RM_scores_all_proteins = df_RM_scores_all_proteins.append(df_RM_scores)
            
            ##### update progress bar
            progress += progress_step
            self.send_progress(progress)


    else:
        
        ##### send status to statusbar
        self.send_status("FLEXIQuant-LF computation with multiprocessing...")
        
        ##### run the FLEXIQuant-LF method (run_FQLF) for all proteins in parallel
        FQLF_res = [] ##### list for RM scores dataframes of multiprocessing

        ##### start multiprocessing by initializing Pools
        pool = Pool(min(self.num_cpus, cpu_count()-1))
        
        ##### compute run_FQLF in parallel
        res = pool.starmap_async(mFLF.run_FQLF, 
                                 [(df_intens_matrix_all_proteins[protein].copy(), protein, str(self.reference), 
                                   self.num_ransac_init, self.mod_cutoff, self.remove_outliers, self.input_file_name, 
                                   self.path_output, reg_plots_pdf, scatter_plots_pdf) for protein in list_proteins], 
                                 callback=FQLF_res.append, chunksize=1)

        ##### close multiprocessing pools
        pool.close()

        ##### update progress every second during FLEXIQuant-LF computation
        while True:
            ##### processing done update progress and break the loop
            if (res.ready()): 
                ##### update progress bar
                #progress = progress_step*num_proteins
                self.send_progress(progress_step*num_proteins)
                break
            ##### get remaining number of proteins for processing
            remaining = res._number_left
            ##### update progress bar
            progress = progress_step*(num_proteins-remaining)
            self.send_progress((progress_step*(num_proteins-remaining)))
            sleep(0.1)
            
        ##### join multiprocessing threads
        pool.join()
        
        ##### get RM scores for all proteins
        for elem in FQLF_res[0]:
            df_RM_scores_all_proteins = df_RM_scores_all_proteins.append(elem)


    ##### close pdf file if plots were created
    if self.create_plots:
        reg_plots_pdf.close()
        scatter_plots_pdf.close()
    
    ##### send status to statusbar
    self.send_status("Finished FLEXIQuant-LF analysis in {:.3f} minutes".format((time()-starttime)/60))
    
    ##### list of all groups for creation the distribution plots and protein-wise heatmaps
    list_groups = list(set(df_RM_scores_all_proteins.columns.get_level_values("Group")))
    list_groups.sort()
    
    ##### create plot of the distribution of RM scores for each sample per sample group
    mFLF.create_RM_score_distribution_plots(df_RM_scores_all_proteins, distri_plots_pdf, list_groups)
    ##### close pdf file of the distribution plots
    try: distri_plots_pdf.close()
    except: pass
    
    ##### define the colormap for the heatmap as specified by the user
    if self.colormap == "red-white-blue":
        color_map = "RdBu"
    elif self.colormap == "pink-white-green":
        color_map = "PiYG"
    elif self.colormap == "purple-white-green":
        color_map = "PRGn"
    elif self.colormap == "brown-white-bluegreen":
        color_map = "BrBG"
    elif self.colormap == "orange-white-purple":
        color_map = "PuOr"
    elif self.colormap == "red-white-grey":
        color_map = "RdGy"
    elif self.colormap == "red-yellow-green":
        color_map = "RdYlGn"
    elif self.colormap == "red-yellow-blue":
        color_map = "RdYlBu"
    
    ##### if choosen create heatmaps for every protein of the RM scores of their peptides
    if self.create_heatmaps:
        
        ##### restart progress from 0 for heatmaps computation
        progress = 0
        self.send_progress(0)
        
        ##### send status to statusbar
        self.send_status("Create heatmaps for every protein...")

        ##### create title page
        fig = plt.figure(figsize=(1, 1))
        plt.title("Heatmaps of RM scores of multiFLEX-LF", fontsize=20)
        plt.axis("off")
        heatmaps_pdf.savefig(figure=fig, bbox_inches="tight", dpi=300)
        plt.close()
        
        ##### define color map for the heatmaps
        custom_cmap = copy(get_cmap(color_map))
        ##### missing values are set to black
        custom_cmap.set_bad("black",1.)
        custom_norm = Normalize(vmin=0, vmax=self.mod_cutoff*2)
        
        ##### sort the proteins descending by number of peptides and samples with a RM scores below the modification cutoff
        sorted_proteins = list(df_RM_scores_all_proteins[df_RM_scores_all_proteins < self.mod_cutoff].count(axis=1).groupby("ProteinID").sum().sort_values(ascending=False).index)
        
        ##### go through all protein in the sorted order
        for protein_id in sorted_proteins:
            
            ##### dataframe of the RM scores of the current protein
            df_RM_scores_protein = df_RM_scores_all_proteins.loc[protein_id]
            
            ##### skip the protein, if dataframe empty
            if df_RM_scores_protein.empty:
                continue
            
            ##### ignore nan-slice warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
                
                ##### create heatmap of the current protein
                mFLF.create_heatmap(df_RM_scores_protein, protein_id, heatmaps_pdf, list_groups, custom_cmap, custom_norm)
            
            ##### update progress bar
            progress += progress_step
            self.send_progress(progress)
            
        ##### close pdf file of the heatmaps
        heatmaps_pdf.close()
        
        ##### send status to statusbar
        self.send_status("Finished with heatmaps for every protein!")


    ##### if chosen compute the clustering of the RM scores 
    ##### and create the html file with the interactive dendrogram and heatmap
    if self.clustering:
        
        ##### update progress bar
        self.send_progress(-1)
        
        ##### send status to statusbar
        self.send_status("Clustering of the peptides based on their RM scores...")

        ##### keep only peotides that have RM scores in at least two groups
        to_remove = df_RM_scores_all_proteins.loc[df_RM_scores_all_proteins.groupby("Group", axis=1).count().replace(0, nan).count(axis=1) < 2].index
        df_RM_scores_all_proteins_reduced = df_RM_scores_all_proteins.drop(to_remove, axis=0)
        removed_peptides = DataFrame(list(to_remove))
        to_remove = DataFrame()
        
        ##### impute missing values for clustering
        df_RM_scores_all_proteins_reduced, df_RM_scores_all_proteins_imputed, removed = mFLF.missing_value_imputation(df_RM_scores_all_proteins_reduced, round(1-self.cosine_similarity, 3))
        removed_peptides = removed_peptides.append(removed)
        removed = DataFrame()
        
        ##### if chosen apply DESeq2 normalization to the RM scores
        if self.deseq_normalization:
            
            ##### send to statusbar
            self.send_status("DESeq2 normalization of the RM scores...")
            
            ##### create sample-group dataframe
            groups = DataFrame(list(df_RM_scores_all_proteins.columns))
            groups.columns = ["Group", "Sample"]
            
            ##### save sample groups as csv file for DESeq2 normalization, 
            ##### raise permission error if file can not be accessed
            groups_file = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_sample_groups.csv")
            try: groups.to_csv(groups_file, sep=',', mode='w', index=False, header=True)
            except PermissionError:
                ##### print error message
                self.send_error_file_open(groups_file)
                ##### terminate process
                exit(ui.worker_thread.exec_())
            
            ##### save RM scores dataframe with imputed values as csv file
            imputed_file = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_RM_scores_imputed.csv")
            try: df_RM_scores_all_proteins_imputed.droplevel("Group", axis=1).to_csv(imputed_file, sep=',', mode='w', index=True, header=True)
            except PermissionError:
                ##### print error message and terminate the process
                print("ERROR: Permission denied!\n" + "Please close " + imputed_file)
                exit()

            try:
                ##### execute the R script for DESeq2
                script = "run_deseq2.R"
                command = ["Rscript", script, imputed_file, groups_file]
                x = run(command)
            
            
                if x.returncode == 0:
                    
                    ##### read DESeq2 output file
                    imputed_file_deseq = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_RM_scores_imputed_DESeq.csv")
                    try: df_RM_scores_all_proteins_imputed = read_csv(imputed_file_deseq, sep=',', header=[0], index_col=[0,1])
                    except PermissionError:
                        ##### print error message
                        self.send_error_file_open(imputed_file_deseq)
                        ##### terminate process
                        exit(ui.worker_thread.exec_())
                    
                    ##### rename columns
                    df_RM_scores_all_proteins_imputed.columns = df_RM_scores_all_proteins_reduced.columns
                    
                    ##### remove the deseq2 input and output file
                    remove(imputed_file)
                    remove(imputed_file_deseq)
                    
                    ##### keep only normalized RM scores which were not missing before the previous imputation
                    ##### then reimpute the missing values
                    df_RM_scores_all_proteins_reduced = df_RM_scores_all_proteins_imputed[df_RM_scores_all_proteins_reduced.isna() == False]
                    df_RM_scores_all_proteins_reduced = round(df_RM_scores_all_proteins_reduced, 5)
                    
                    ##### impute missing values again
                    df_RM_scores_all_proteins_reduced, df_RM_scores_all_proteins_imputed, removed = mFLF.missing_value_imputation(df_RM_scores_all_proteins_reduced, round(1-self.cosine_similarity, 5))
                    ##### dataframe of peptides that were removed during imputation
                    removed_peptides = removed_peptides.append(removed)
                    removed = DataFrame()
                
                    ##### write DESeq2 normalized RM scores to csv file
                    imputed_file = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_RM_scores_DESeq.csv")
                    try: df_RM_scores_all_proteins_reduced.droplevel("Group", axis=1).to_csv(imputed_file, sep=',', mode='w', index=True, header=True)
                    except PermissionError:
                        ##### print error message
                        self.send_error_file_open(imputed_file)
                        ##### terminate process
                        exit(ui.worker_thread.exec_())
                    
                    
                    ##### create plot of the distribution of RM scores for each sample per sample group
                    mFLF.create_RM_score_distribution_plots(df_RM_scores_all_proteins_reduced, deseq_distri_plots_pdf, list_groups)
                    ##### close pdf file of the distribution plots
                    try: deseq_distri_plots_pdf.close()
                    except: pass
                                    
                    ##### output files for the heatmap and csv file of the clustering
                    clust_heatmap = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_DESeq_clustered_heatmap.html")
                    output_df_file = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_DESeq_clustered.csv")
                
                else: 
                    
                    ##### if DESeq2 normalization did not work, show warning
                    ##### and continue without normalized RM scores
                    self.send_warning_deseq()
                    
                    ##### remove the deseq2 input file
                    remove(imputed_file)
                    
                    ##### output files for the heatmap and csv file of the clustering
                    clust_heatmap = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_clustered_heatmap.html")
                    output_df_file = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_clustered.csv")
            
            except:
                
                ##### if DESeq2 normalization did not work, show warning
                ##### and continue without normalized RM scores
                self.send_warning_plots()
                
                ##### remove the deseq2 input file
                remove(imputed_file)
                
                ##### output files for the heatmap and csv file of the clustering
                clust_heatmap = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_clustered_heatmap.html")
                output_df_file = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_clustered.csv")
    
        else:
            
            ##### output files for the heatmap and csv file of the clustering
            clust_heatmap = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_clustered_heatmap.html")
            output_df_file = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF_RM_scores_clustered.csv")
        
        ##### send status to statusbar
        self.send_status("Clustering of the peptides based on their RM scores...")
        
        ##### add removed peptides from imputation to csv file
        #print(len(removed_peptides), "peptides removed during imputation")
        if len(removed_peptides) > 0:
            removed_peptides.columns = ["ProteinID", "PeptideID"]
            removed_peptides = removed_peptides.set_index(["ProteinID"])
            
            ##### save removed peptides in ...mFQ-LF-output_removed_peptides.csv file, raise permission error if file can not be accessed
            path_out = mFLF.add_filename_to_path(self.path_output, self.input_file_name, "mFQ-LF-output_removed_peptides.csv")
            try: removed_peptides.to_csv(path_out, sep=',', mode='a', index=True, header=False)
            except PermissionError:
                ##### print error message
                self.send_error_file_open(path_out)
                ##### terminate process
                exit(ui.worker_thread.exec_())
                
                
        ##### begin clustering #####
        ##### cluster the peptide by their RM scores over the samples
        ##### calculate linkage matrix for all peptides in the dataframe 
        linkage_matrix = linkage(df_RM_scores_all_proteins_imputed, 
                                 metric = lambda u, v: mFLF.RM_score_distance(u, v, self.mod_cutoff), 
                                 method = "average")
        
        ##### create plotly figure
        fig, array_RM_scores_all_proteins_reduced, ordered_peptides = mFLF.peptide_clustering(df_RM_scores_all_proteins_reduced, linkage_matrix, self.mod_cutoff, color_map, ["black"]*8, None, [])
        
        ##### define options to edit plotly figure in the browser and save a screenshot of the figure
        plotly_config = {"displayModeBar": True, 
                         "editable": True,
                         "toImageButtonOptions": {
                             "format": "png",
                             "filename": clust_heatmap.split('/')[-1][:-5],
                             "scale": 2
                             }
                         }
        
        ##### open the plotly figure in the system's default internet browser
        fig.show(config=plotly_config)
        
        ##### write plotly figure to the html file
        try: fig.write_html(file=clust_heatmap, include_plotlyjs=True, full_html=True, config=plotly_config)
        except PermissionError:
            ##### print error message
            self.send_error_file_open(clust_heatmap)
            ##### terminate process
            exit(ui.worker_thread.exec_())
        
        ##### create output of the RM scores in same order as in the heatmap
        output_df = DataFrame(flip(array_RM_scores_all_proteins_reduced, axis=0))
        output_df.columns = df_RM_scores_all_proteins_reduced.columns.get_level_values("Sample")
        ##### add the ID column
        output_df.index = MultiIndex.from_tuples(flip(array(df_RM_scores_all_proteins_reduced.index)[ordered_peptides]), names =("ProteinID", "PeptideID"))
        output_df = output_df.reset_index()
        output_df.index.names = ["ID"]
        ##### write the output to a csv file
        try: output_df.to_csv(output_df_file, sep=',', mode='w', index=True, header=True, float_format="%.5f")
        except PermissionError:
            ##### print error message
            self.send_error_file_open(output_df_file)
            ##### terminate process
            exit(ui.worker_thread.exec_())
        
        ##### get system time
        endtime = time()
        
        ##### flat cluster creation from the hierarchical clustering #####
        ##### show input prompt for the user defined distance of the hierachical clustering which should be used for generation of clusters
        ##### redo plotly figure with a colored dendrogram by the defined distance threshold
        ##### save plotly figure in html and create new output with cluster ids
        
        ##### list of colors for the dendrogram
        colors_list = ['rgb'+str(elem) for elem in color_palette("Set2", 8)]
        
        ##### send signal to open an input dialog and get the return value
        distance_str = str(self.open_input_dialog())
        
        ##### until Cancel is pressed reapeat opening the input window 
        ##### and ask for a clutering distance cutoff
        while distance_str != 'q':
            
            ##### if Cancel was pressed finish multiFLEX-LF computation
            if distance_str == "q":
                ##### send status to statusbar
                self.send_status("Finished clustering!")
                
                ##### send status to statusbar
                self.send_status("Finished with multiFLEX-LF analysis in {:.3f} minutes".format((endtime-starttime)/60))

            ##### get absolute float value of the distance
            distance_int = abs(float(distance_str))
            
            ##### create the flat clusters based on the given distance cutoff 
            ##### and create an array of the cluster ids
            ##### has to be flipped because the dendrogram and heatmap is flipped
            array_cluster_ids = flip(fcluster(linkage_matrix, t=distance_int, criterion="distance")[ordered_peptides])
            
            ##### create plotly figure with the distance threshold
            fig, array_RM_scores_all_proteins_reduced, ordered_peptides = mFLF.peptide_clustering(df_RM_scores_all_proteins_reduced, linkage_matrix, self.mod_cutoff, color_map, colors_list, distance_int, array_cluster_ids)
            ##### open the plotly figure in the system's default internet browser
            fig.show(config=plotly_config)
            
            ##### write plotly figure to the html file
            try: fig.write_html(file=clust_heatmap[:-5]+"_dist-"+str(distance_int)+".html", include_plotlyjs=True, full_html=True, config=plotly_config)
            except PermissionError:
                ##### print error message
                self.send_error_file_open(clust_heatmap[:-5]+"_dist-"+str(distance_int)+".html")
                ##### terminate process
                exit(ui.worker_thread.exec_())
            
            ###### add Cluster ids to the output table and print the dataframe to a new file
            output_df["Cluster"] = array_cluster_ids
            
            try: output_df.to_csv(output_df_file[:-4]+"_dist-"+str(distance_int)+".csv", sep=',', mode='w', index=True, header=True, float_format="%.5f")
            except PermissionError:
                ##### print error message
                self.send_error_file_open(output_df_file[:-4]+"_dist-"+str(distance_int)+".csv")
                ##### terminate process
                exit(ui.worker_thread.exec_())
            
            
            ##### again ask for user chosen distance cutoff for the clusters
            distance_str = str(self.open_input_dialog())
            
        
        ##### send status to statusbar
        self.send_status("Finished clustering!")
    
    ##### send status to statusbar
    self.send_status("Finished with multiFLEX-LF analysis in {:.3f} minutes".format((endtime-starttime)/60))
    
    ##### update progress bar to finish computation
    self.send_progress(100)



##### main script #####
if __name__ == "__main__":
    
    freeze_support() ##### support of multiprocessing for executable creation
    
    ##### initiate QApplication
    app = QtWidgets.QApplication(argv)
    app.setStyle('Fusion')
    #app.setFont(QFont('Ariel', 10))
    
    ##### initiate main window
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()

    ##### set up main window
    ui.setupUi(MainWindow)

    ##### display main window
    MainWindow.show()

    ##### display exit code
    exit(app.exec_())

