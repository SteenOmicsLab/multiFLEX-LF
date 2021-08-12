#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors: Pauline Hiort and Konstantin Kahnert
# Date: 2021_07_22
# Python version: 3.8

"""
multiFLEX-LF CLI version

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
    - input_file: path to .csv file containing peptide intensities (str)
    - output_folder: path of the folder into which the output files should be written (str)
    - reference: identifier of the reference sample(s) (str)
    - num_init: number of ransac initiations (int)
    - mod_cutoff: RM score cutoff used to classify peptides as differentially modified (float between 0 and 1)
    - create_plots: if True: regression and scatter plots are created; 
                    if False: no plots are created
    - create_heatmaps: if True: for every protein a heatmap of the RM scores is created
    - remove_outliers: if True: peptides with a row score above a threshold are removed;
                       if False: only outlier raw score is removed and not the peptide
    - num_cpus: number of cpus/threads used for multiFLEX-LF computation (int)
    - clustering: if True: compute hierarchical clustering of the peptides
    - cosine_similarity: similarity threshold for missing RM score imputation (float between 0 and 1)
    - deseq2_normalization: if True: apply deseq2 normalization of the RM scores (requires R and DESeq2 installation!)
    - colormap: colormap used for creating the heatmaps 
"""


##### import of the required libraries #####

##### import multiFLEX-LF methods
import multiFLEX_LF_methods as mFLF

##### used  for command line options
import click

##### used to exit multiFLEX-LF 
from sys import exit

##### used for directory and file manipulation
from os import mkdir, remove
from os.path import isdir
from subprocess import run

##### used for regular expressions
from re import findall

##### used for data manipulation
from pandas import read_csv, Series, DataFrame, MultiIndex
from numpy import nan, array, square, sqrt, warnings, isnan, ones, arange, flip, log10

##### used for parallel computation
from multiprocessing import Pool, freeze_support, cpu_count

##### used for runtime tracking
from time import time

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
##### open plotly figures in default internet browser
renderers.default = "browser"


##### definition of the command-line input parameters #####
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
 
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    '-i',
    '--input_file',
    type=click.Path(exists=True),
    required=True,
    help='Path of the comma separated input file (.csv)'
)
@click.option(
    '-o',
    '--output_folder',
    type=click.Path(exists=True),
    required=True,
    help='Path to the folder into which the output files should be saved'
)
@click.option(
    '-r',
    '--reference',
    required=True,
    help='Reference sample identifier. This must match the Group column value of the reference samples.'
)
@click.option(
    '-n',
    '--num_init',
    default=30,
    type=click.IntRange(min=5, max=100),
    help='Number of times multiFLEX-LF/FLEXIQuant-LF fits a new RANSAC linear regression model to each sample to choose the best '
         'model. Should be an integer between 5 and 100. The more initiations, the higher the reproducibility '
         'and the probability the optimal model is found. However, choosing a high number can significantly '
         'increases the run time and more than 50 initiations rarely provide additional benefit. '
         'The default value is: 30'
)
@click.option(
    '-mc',
    '--mod_cutoff',
    default=0.5,
    type=click.FloatRange(min=0, max=1),
    help='RM score cutoff used to classify peptides as differentially modified. Should be a a float between 0 and 1. '
         'The default value is: 0.5'
)
@click.option(
    '-rem',
    '--remove_outliers',
    is_flag=True,
    help='If selected the peptides that are considered as outlier based on their raw scores are removed before RM score calculation. '
         'Otherwise only the raw score that is considered an outlier is removed before RM score calculation. '
)
@click.option(
    '-p',
    '--create_plots',
    is_flag=True,
    help='If selected a linear regression plots for each sample and a scatter plot of the peptides intensities vs the RM scores are created for each proteins. '
         'All plots will be saved in a pdf file. Warning: This option only works with 1 CPU core! '
)

@click.option(
    '-hmap',
    '--create_heatmaps',
    is_flag=True,
    help='If selected a heatmap of the RM scores is created for every protein. '
         'All heatmaps will be saved in one pdf file. '
)
@click.option(
    '-cpu',
    '--num_cpus',
    default=1,
    type=click.IntRange(min=1),
    help='Number of threads/CPUs employed for multiFLEX-LF calculation. '
         'Warning: This number should be lower than the maximum number of available threads, otherwise the computer might freeze! '
)
@click.option(
    '-c',
    '--clustering',
    is_flag=True,
    help='If selected hierachical clustering of the RM scores of all peptides in the dataset is computed. '
         'The clustring is saved as a interactive dendrogram and heatmap in a html-file. '
)
@click.option(
    '-cos',
    '--cosine_similarity',
    default=0.98,
    type=click.FloatRange(min=0, max=1),
    help='Minimal cosine similarity value for missing value imputation of RM scores for clustering. '
         'Should be a a float between 0 and 1. The default value is: 0.98 '
)
@click.option(
    '-dn',
    '--deseq2_normalization',
    is_flag=True,
    help='If selected RM scores are normalized with DESeq2 before clustering. '
         'Requires a working installation of R and DESeq2! '
)
@click.option(
    '-cm',
    '--colormap',
    default=1,
    type=click.IntRange(min=1, max=7),
    help='Colormap used for the heatmap(s). Please choose one of 1-7. Value description: \b\n\n'
         '1 = "red-white-blue" colormap \b\n 2 = "pink-white-green" \b\n 3 = "purple-white-green"'
         '\b\n 4 = "brown-white-bluegreen" \b\n 5 = "orange-white-purple" \b\n 6 = "red-white-grey"'
         '\b\n 7 = "red-yellow-green" \b\n 8 = "red-yellow-blue; \b\n See: https://matplotlib.org/stable/tutorials/colors/colormaps.html"'
)



def mFQLF_CLI_main(input_file, output_folder, reference, num_init, mod_cutoff, remove_outliers, 
                  create_plots, create_heatmaps, clustering, cosine_similarity, deseq2_normalization,
                  num_cpus, colormap):
    """
    multiFLEX-LF CLI

    Usage example:
    python multiFLEX_LF_CLI.py -i path/to/input_file.csv -o path/to/output_folder/ -r "Control" -n 30 -mc 0.5 -p -hmap -cpu 1 -c -cos 0.98 -dn -cm 1

    """
    
    #### get current system time to track the runtime
    starttime = time()
    
    ##### print status
    print("Starting multiFLEX-LF/FLEXIQuant-LF analysis...")
        
    ##### get input file name
    input_file_name = findall("[^\\\\,/]*$", input_file)[0]
    
    ##### append trailing '/' to the output_folder string if missing
    if output_folder == '':
        input_file = input_file.replace('\\', '/')
        output_folder =  '/'.join(input_file.split('/')[:-1])+'/'
    
    ##### load data into pandas dataframes
    try: df_input = read_csv(input_file, sep=',', index_col=None)
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + input_file)
        exit()
    
    ##### check if ProteinID, PeptideID, Group, Sample, Intensity columns exists
    try:
        df_input["ProteinID"]
        df_input["PeptideID"]
        df_input["Group"]
        df_input["Sample"]
        df_input["Intensity"]
    except KeyError:
        ##### print error message and terminate the process
        print("ERROR: Incorrect input format!\n" + "At least one of the columns \"ProteinID\", \"PeptideID\", \"Sample\", \"Group\" or \"Intensity\" could not be found in: "+input_file)
        exit()

    ##### check if given reference identifier exists in Group column
    if str(reference) not in set(df_input["Group"].astype(str)):
        ##### print error message and terminate the process
        print("ERROR: Given reference sample identifier \"" + str(reference) + "\" not found in \"Group\" column")
        exit()
        
    ##### create the output directory if it does not exist
    if not isdir(output_folder): mkdir(output_folder)

    ##### create the intensities matrix
    ##### ProteinID and PeptidesID as column indices; Group and Sample as row indices
    df_intens_matrix_all_proteins = df_input.set_index(["ProteinID", "PeptideID", "Group", "Sample"]).unstack(level=["Group", "Sample"]).T
    df_intens_matrix_all_proteins = df_intens_matrix_all_proteins.set_index([df_intens_matrix_all_proteins.index.get_level_values("Group"), df_intens_matrix_all_proteins.index.get_level_values("Sample")])
    df_intens_matrix_all_proteins = df_intens_matrix_all_proteins.sort_index(axis=0).sort_index(axis=1)
        
    ##### check if regression and scatter plots should and can be created
    ##### construction of the regression and scatter plots does not work with multiprocessing!
    if create_plots and num_cpus < 2:
        
        ##### create output file path for the pdf file of the regression plots
        path_regression_plots = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_regression_plots.pdf")

        ##### try to access the pdf and raise permission error if not possible
        try: reg_plots_pdf = PdfPages(path_regression_plots)
        except PermissionError:
            ##### print error message and terminate the process
            print("ERROR: Permission denied!\n" + "Please close " + path_regression_plots)
            exit()
        
        ##### create title page for regression plots
        fig = plt.figure(figsize=(1, 1))
        plt.title("RANSAC Linear Regression Plots of multiFLEX-LF/FLEXIQuant-LF", fontsize=20)
        plt.axis("off")
        reg_plots_pdf.savefig(figure=fig, bbox_inches="tight", dpi=300)
        plt.close()
        
        
        ##### create output file path for the pdf file of the intensities vs. RM scores scatter plot
        path_scatter_plots = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_scatter_plots.pdf")
        ##### try to access the pdf file and raise permission error if not possible
        try: scatter_plots_pdf = PdfPages(path_scatter_plots)
        except PermissionError:
            ##### print error message and terminate the process
            print("ERROR: Permission denied!\n" + "Please close " + path_scatter_plots)
            exit()
        
        ##### create title page for scatter plots
        fig = plt.figure(figsize=(1, 1))
        plt.title("Peptide Intensity vs. RM score Scatter Plots of multiFLEX-LF", fontsize=20)
        plt.axis("off")
        scatter_plots_pdf.savefig(figure=fig, bbox_inches="tight", dpi=300)
        plt.close()
        
    else:
        
        ##### if multiprocessing is activated print warning
        if create_plots:
            print("WARNING: Can not create regression and scatter plots with multiprocessing!\nPlease use -cpu 1 to create regression and scatter plots!\nContinuing without creating the plots.")
            print()
            
        ##### creating the plots does not work with multiprocessing!
        ##### turn it off and continue without plot creation
        create_plots = False
        reg_plots_pdf = ''
        scatter_plots_pdf = ''
    
    
    ##### create output file path for the pdf file of the RM score distribution
    path_distribution_plots = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_distribution.pdf")
    ##### try to access the pdf file and raise permission error if not possible
    try: distri_plots_pdf = PdfPages(path_distribution_plots)
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_distribution_plots)
        exit()
    
    ##### check if heatmaps should and can be created
    if create_heatmaps:
        path_heatmaps = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_heatmaps.pdf")
        ##### try to access the pdf file and raise permission error if not possible
        try: heatmaps_pdf = PdfPages(path_heatmaps)
        except PermissionError:
            ##### print error message and terminate the process
            print("ERROR: Permission denied!\n" + "Please close " + path_heatmaps)
            exit()
    
    ##### create header for csv output files
    header = df_intens_matrix_all_proteins.T.droplevel("Group", axis=1).head(0)
    
    ##### save raw scores as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_raw_scores.csv")
    try: header.to_csv(path_out, sep=',', mode='w', index=True, header=True, float_format="%.5f")
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()

    ##### save RM scores as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_RM_scores.csv")
    try: header.to_csv(path_out, sep=',', mode='w', index=True, header=True, float_format="%.5f")
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()

    ##### save differentially modified dataframe as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_diff_modified.csv")
    try: header.to_csv(path_out, sep=',', mode='w', index=True, header=True)
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()
    
    ##### create header for output file containing removed peptides
    removed_header = DataFrame([{"ProteinID": 0, "PeptideID": 0}])
    
    ##### save removed peptides as csv file, raise permission error if file can not be accessed
    path_out = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_removed_peptides.csv")
    try: removed_header.head(0).to_csv(path_out, sep=',', mode='w', index=False, header=True)
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()
    
    ##### remove the header variables
    del header, removed_header
    
    ##### start multiFLEX-LF/FLEXIQuant-LF computation #####
    
    ##### dataframe for the calculated RM scores of all proteins and peptides
    df_RM_scores_all_proteins = DataFrame()
    
    ##### create a list of all proteins in the dataset
    list_proteins = df_intens_matrix_all_proteins.columns.get_level_values("ProteinID").unique().sort_values()      
    
    ##### print status
    print("Data was imported sucessfully")
    print("Analyzing", len(list_proteins), "proteins...")    
    
    ##### if number of cpus smaller than 2 continue without parallel processing
    ##### otherwise use parallel processing to analyze all proteins
    if num_cpus < 2:
        
        ##### print status
        print("FLEXIQuant-LF computation...")
        
        for protein in list_proteins:
  
            ##### run the FLEXIQuant-LF method (run_FQLF) with the current protein
            df_RM_scores = mFLF.run_FQLF(df_intens_matrix_all_proteins[protein].copy(), protein, str(reference), 
                                    num_init, mod_cutoff, remove_outliers, input_file_name, output_folder, 
                                    reg_plots_pdf, scatter_plots_pdf)
            
            ##### add RM scores of protein to dataframe of all proteins
            df_RM_scores_all_proteins = df_RM_scores_all_proteins.append(df_RM_scores)

    else:
        
        ##### print status
        print("FLEXIQuant-LF computation with multiprocessing...")
        
        ##### list of the RM scores of run_FQLF() while multiproccessing
        FQLF_res = []
        
        ##### start multiprocessing by initializing Pools
        pool = Pool(min(num_cpus, cpu_count()-1))
        
        ##### run the FLEXIQuant-LF method (run_FQLF) with the current protein
        ##### compute run_FQLF in parallel
        pool.starmap_async(mFLF.run_FQLF, 
                           [(df_intens_matrix_all_proteins[protein].copy(), protein, str(reference), 
                             num_init, mod_cutoff, remove_outliers, input_file_name, output_folder, 
                             reg_plots_pdf, scatter_plots_pdf) for protein in list_proteins], 
                           callback=FQLF_res.append)
        
        ##### close multiprocessing pools
        pool.close()
        pool.join()
        
        ##### get RM scores for all proteins
        for elem in FQLF_res[0]:
            df_RM_scores_all_proteins = df_RM_scores_all_proteins.append(elem)
    
    
    ##### close pdf file if plots were created
    if create_plots:
        reg_plots_pdf.close()
        scatter_plots_pdf.close()
    
    ##### print status
    print("Finished FLEXIQuant-LF analysis in {:.3f} minutes".format((time()-starttime)/60))
    
    ##### list of all groups for creation the distribution plots and protein-wise heatmaps
    list_groups = list(set(df_RM_scores_all_proteins.columns.get_level_values("Group")))
    list_groups.sort()
    
    ##### create plot of the distribution of RM scores for each sample per sample group
    mFLF.create_RM_score_distribution_plots(df_RM_scores_all_proteins, distri_plots_pdf, list_groups)
    ##### close pdf file of the distribution plots
    try: distri_plots_pdf.close()
    except: pass
    
    ##### define the colormap for the heatmap as specified by the user
    if colormap == 1:
        color_map = "RdBu"
    elif colormap == 2:
        color_map = "PiYG"
    elif colormap == 3:
        color_map = "PRGn"
    elif colormap == 3:
        color_map = "BrBG"
    elif colormap == 4:
        color_map = "PuOr"
    elif colormap == 5:
        color_map = "RdGy"
    elif colormap == 6:
        color_map = "RdYlGn"
    elif colormap == 7:
        color_map = "RdYlBu"
    
    ##### check if colormap is valid
    try: custom_cmap = copy(get_cmap(color_map))
    except ValueError:
        print("ERROR: Invalid color map!\n Please choose a valid color map. See: https://matplotlib.org/stable/tutorials/colors/colormaps.html")
        exit()
    
    ##### if choosen create heatmaps for every protein of the RM scores of their peptides
    if create_heatmaps:
        
        ##### print status
        print("Creating heatmaps for every protein...")
        
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
        custom_norm = Normalize(vmin=0, vmax=mod_cutoff*2)
        
        ##### sort the proteins descending by number of peptides and samples with a RM scores below the modification cutoff
        sorted_proteins = list(df_RM_scores_all_proteins[df_RM_scores_all_proteins < mod_cutoff].count(axis=1).groupby("ProteinID").sum().sort_values(ascending=False).index)
        
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
        
        ##### close pdf file of the heatmaps
        heatmaps_pdf.close()
        
        
    ##### if chosen compute the clustering of the RM scores 
    ##### and create the html file with the interactive dendrogram and heatmap
    if clustering:
        
        ##### print status
        print("Clustering of the peptides based on their RM scores...")

        ##### keep only peptides that have RM scores in at least two groups
        to_remove = df_RM_scores_all_proteins.loc[df_RM_scores_all_proteins.groupby("Group", axis=1).count().replace(0, nan).count(axis=1) < 2].index
        df_RM_scores_all_proteins_reduced = df_RM_scores_all_proteins.drop(to_remove, axis=0)
        removed_peptides = DataFrame(list(to_remove))
        to_remove = DataFrame()
        
        ##### impute missing values for clustering
        df_RM_scores_all_proteins_reduced, df_RM_scores_all_proteins_imputed, removed = mFLF.missing_value_imputation(df_RM_scores_all_proteins_reduced, round(1-cosine_similarity, 3))
        removed_peptides = removed_peptides.append(removed)
        removed = DataFrame()
        
        ##### if chosen apply DESeq2 normalization to the RM scores
        if deseq2_normalization:
            
            ##### print status
            print("Applying DESeq2 normalization of the RM scores...")
            
            ##### create sample-group dataframe
            groups = DataFrame(list(df_RM_scores_all_proteins.columns))
            groups.columns = ["Group", "Sample"]
            
            ##### save sample groups as csv file for DESeq2 normalization, 
            ##### raise permission error if file can not be accessed
            groups_file = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_sample_groups.csv")
            try: groups.to_csv(groups_file, sep=',', mode='w', index=False, header=True)
            except PermissionError:
                ##### print error message and terminate the process
                print("ERROR: Permission denied!\n" + "Please close " + groups_file)
                exit()
            
            ##### save RM scores dataframe with imputed values as csv file
            imputed_file = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_RM_scores_imputed.csv")
            try: df_RM_scores_all_proteins_imputed.droplevel("Group", axis=1).to_csv(imputed_file, sep=',', mode='w', index=True, header=True)
            except PermissionError:
                ##### print error message and terminate the process
                print("ERROR: Permission denied!\n" + "Please close " + imputed_file)
                exit()

            try:
                ##### execute the R script for DESeq2
                script = "run_deseq2.R"
                cmd = ["Rscript", script, imputed_file, groups_file]
                x = run(cmd)
                
                if x.returncode == 0:
                    
                    ##### read DESeq2 output file
                    imputed_file_deseq = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_RM_scores_imputed_DESeq.csv")
                    try: df_RM_scores_all_proteins_imputed = read_csv(imputed_file_deseq, sep=',', header=[0], index_col=[0,1])
                    except PermissionError:
                        ##### print error message and terminate the process
                        print("ERROR: Permission denied!\n" + "Please close " + imputed_file_deseq)
                        exit()
                    
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
                    df_RM_scores_all_proteins_reduced, df_RM_scores_all_proteins_imputed, removed = mFLF.missing_value_imputation(df_RM_scores_all_proteins_reduced, round(1-cosine_similarity, 5))
                    ##### dataframe of peptides that were removed during imputation
                    removed_peptides = removed_peptides.append(removed)
                    removed = DataFrame()
                    
                    ##### write DESeq2 normalized RM scores to csv file
                    imputed_file = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_RM_scores_DESeq.csv")
                    try: df_RM_scores_all_proteins_reduced.droplevel("Group", axis=1).to_csv(imputed_file, sep=',', mode='w', index=True, header=True)
                    except PermissionError:
                        ##### print error message and terminate the process
                        print("ERROR: Permission denied!\n" + "Please close " + imputed_file)
                        exit()
                    
                    ##### output files for the heatmap and csv file of the clustering
                    clust_heatmap = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_DESeq_clustered_heatmap.html")
                    output_df_file = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_DESeq_clustered.csv")
                
                else:
                    
                    ##### if DESeq2 normalization did not work, print warning
                    print("WARNING: DESeq2 Normalization did not work. Continuing without.")
                    print()
                    
                    ##### remove the deseq2 input file
                    remove(imputed_file)
                    
                    ##### output files for the heatmap and csv file of the clustering
                    clust_heatmap = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_clustered_heatmap.html")
                    output_df_file = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_clustered.csv")
                    
            except:
                
                ##### if DESeq2 normalization did not work, print warning 
                ##### and continue without normalized RM scores
                print("WARNING: DESeq2 Normalization did not work. Continuing without.")
                print()
                
                ##### remove the DESeq2 input file
                remove(imputed_file)
                
                ##### output files for the heatmap and csv file of the clustering
                clust_heatmap = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_clustered_heatmap.html")
                output_df_file = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_clustered.csv")
                
        else:
            
            ##### output files for the heatmap and csv file of the clustering
            clust_heatmap = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_clustered_heatmap.html")
            output_df_file = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF_RM_scores_clustered.csv")
        
        ##### add removed peptide to csv file
        #print(len(removed_peptides), "peptides removed during imputation")
        if len(removed_peptides) > 0:
            removed_peptides.columns = ["ProteinID", "PeptideID"]
            removed_peptides = removed_peptides.set_index(["ProteinID"])
            
            ##### save removed peptides, raise permission error if file can not be accessed
            path_out = mFLF.add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_removed_peptides.csv")
            try: removed_peptides.to_csv(path_out, sep=',', mode='a', index=True, header=False)
            except PermissionError:
                ##### print error message and terminate the process
                print("ERROR: Permission denied!\n" + "Please close " + path_out)
                exit()
        
        ##### begin clustering #####
        ##### cluster the peptide by their RM scores over the samples
        ##### calculate linkage matrix for all peptides in the dataframe 
        linkage_matrix = linkage(df_RM_scores_all_proteins_imputed, 
                                 metric = lambda u, v: mFLF.RM_score_distance(u, v, mod_cutoff), 
                                 method = "average")
        
        ##### create plotly figure
        fig, array_RM_scores_all_proteins_reduced, ordered_peptides = mFLF.peptide_clustering(df_RM_scores_all_proteins_reduced, linkage_matrix, mod_cutoff, color_map, ["black"]*8, None, [])
        
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
            ##### print error message and terminate
            print("ERROR: Permission denied!\n" + "Please close " + clust_heatmap)
            exit()
        
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
            ##### print error message and terminate
            print("ERROR: Permission denied!\n" + "Please close " + output_df_file)
            exit()
        
        ##### get system time
        endtime = time()
        
        ##### flat cluster creation from the hierarchical clustering #####
        ##### show input prompt for the user defined distance of the hierachical clustering which should be used for generation of clusters
        ##### redo plotly figure with a colored dendrogram by the defined distance threshold
        ##### save plotly figure in html and create new output with cluster ids
        
        ##### list of colors for the dendrogram
        colors_list = ['rgb'+str(elem) for elem in color_palette("Set2", 8)]
        
        ##### ask for user chosen distance cutoff for the clusters
        print("Type a clustering distance cutoff for building of the clusters and hit enter. \nIf not type q and hit enter.")
        distance_str = input("Type the number (with decimal point) here: ")
        
        ##### repeat show input prompt until q is entered
        while distance_str != 'q':
            
            ##### if q is given finish multiFLEX-LF computation
            if distance_str == "q":
                ##### print status and exit
                print("Finished clustering!")
                print("Finished with multiFLEX-LF analysis in {:.3f} minutes".format((endtime-starttime)/60))
                exit()            
            
            ##### test if input can be converted to float
            try:
                ##### get absolute float value of the distance
                distance_int = abs(float(distance_str))
                
                ##### create the flat clusters based on the given distance cutoff 
                ##### and create an array of the cluster ids
                ##### has to be flipped because the dendrogram and heatmap is flipped
                array_cluster_ids = flip(fcluster(linkage_matrix, t=distance_int, criterion="distance")[ordered_peptides])
                
                ###### create plotly figure with the distance threshold
                fig, array_RM_scores_all_proteins_reduced, ordered_peptides = mFLF.peptide_clustering(df_RM_scores_all_proteins_reduced, linkage_matrix, mod_cutoff, color_map, colors_list, distance_int, array_cluster_ids)
                ##### open the plotly figure in the system's default internet browser
                fig.show(config=plotly_config)
                
                ##### write plotly figure to the html file
                try: fig.write_html(file=clust_heatmap[:-5]+"_dist-"+str(distance_int)+".html", include_plotlyjs=True, full_html=True, config=plotly_config)
                except PermissionError:
                    ##### print error message and terminate
                    print("ERROR: Permission denied!\n" + "Please close " + clust_heatmap[:-5]+"_dist-"+str(distance_int)+".html")
                    exit()
                
                ###### add cluster ids to the output table and print the dataframe to a new file
                output_df["Cluster"] = array_cluster_ids
                
                try: output_df.to_csv(output_df_file[:-4]+"_dist-"+str(distance_int)+".csv", sep=',', mode='w', index=True, header=True, float_format="%.5f")
                except PermissionError:
                    ##### print error message and terminate
                    print("ERROR: Permission denied!\n" + "Please close " + output_df_file[:-4]+"_dist-"+str(distance_int)+".csv")
                    exit()
                    
            except:
                print('Not a valid number! Please try again. For exiting type q and hit enter.')
            
            ##### ask for user chosen distance cutoff for the clusters
            print("Type a clustering distance cutoff for building of the clusters and hit enter. \nIf not type q and hit enter.")
            distance_str = input("Type the number (with decimal point) here: ")
        
        ##### print status
        print("Finished clustering!")
            

    ##### print status
    print("Finished with multiFLEX-LF analysis in {:.3f} minutes".format((endtime-starttime)/60))


if __name__ == "__main__":
    ##### run main
    mFQLF_CLI_main()
