#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors: Pauline Hiort and Konstantin Kahnert
# Date: 2021_07_22
# Python version: 3.8.10

"""
Methods of multiFLEX-LF
"""

##### import of the required libraries #####
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
from multiprocessing import Pool, cpu_count

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


##### definition of functions utilized in multiFLEX-LF (and adapted FLEXIQuant-LF) #####

def add_filename_to_path(path, filename, ending):
    """
    Takes path of input file, removes file extension and adds new filename
    Parameters:
        path : path to input file (string)
        filename : name of new file (string)
        ending : file type e.g. .csv (string)

    Returns:
        input path_filename (string)
        
    Adapted from FLEXIQuant-LF
    """
    ##### match everything until '.' in the path
    filename_start = findall("^(.*?)\..*", filename)[0]

    ##### add new filename with '_' in between
    new_filename = filename_start + '_' + ending

    ##### add new_filename to path_output
    path_out = path + '/' + new_filename

    return path_out



def calculate_confidence_band(slope, median_intens, df_train, X, y, alpha):
    """
    Calculates the confidence bands arround the regression line 
    for the regression plots

    Parameters:
        slope : slope of the regression line
        median_intens : pandas series of the median intensity of the reference samples
        df_train : dataframe of the current sample and the reference intensities per peptide
        X : X data of the RANSAC training 
        y : y data of the RANSAC training
        alpha : alpha for the confidence band calculation

    Returns:
        df_train : updated dataframe with confidence band
    """

    ##### calculate predicted intensity with reference intensity of a peptide 
    ##### and slope of the sample (Y hat)
    Y_pred = slope * median_intens

    ##### calculate W
    N = len(df_train)
    F = f.ppf(q=1 - alpha, dfn=2, dfd=N - 1)
    W = sqrt(2 * F)

    ##### calculate standard deviation (s(Y hat))
    ##### calculate prediction error
    error = y - Y_pred
    error.dropna(inplace=True)

    ##### calculate mean squared error
    MSE = sum(error.apply(square)) / (N - 1)

    ##### calculate mean X intensity
    X_bar = df_train["Reference intensity"].mean()

    ##### iterate over all peptides of the sample
    CB_low = []
    CB_high = []

    ##### iterate through median peptide intensities
    for idx, elem in df_train["Reference intensity"].iteritems():
        ##### calculate squared distance to mean X (numerator)
        dist_X_bar = square(elem - X_bar)

        ##### calculate sum of squared distances to mean X (denominator)
        sum_dist_X_bar = sum(square(X - X_bar))

        ##### calculate standard deviation
        s = float(sqrt(MSE * ((1 / N) + (dist_X_bar / sum_dist_X_bar))))

        ##### calculate predicted intensity for given X
        Y_hat = slope * elem

        ##### calculate high and low CB values and append to list
        cb_low = Y_hat - W * s
        cb_high = Y_hat + W * s

        CB_low.append(cb_low)
        CB_high.append(cb_high)

    ##### add CBs as columns to df_train
    df_train["CB low"] = CB_low
    df_train["CB high"] = CB_high

    return df_train



def create_regression_plots(ax0, ax1, df_train, sample, protein_id, r2_model, 
                            r2_data, slope, alpha):
    """
    Creates a histogram and a scatter plot with regression line and confidence bands
    of a current sample in the given figure axis ax0 and ax1

    Parameters:
        ax0 : subfigure axis for the histogram
        ax1 : subfigure axis for the scatter plot
        df_train : dataframe of the current sample and the reference intensities per peptide
        sample_id : currently analyzed sample
        protein_id : id of the currently analyzed protein
        r2_model : r2 score of the model
        r2_data : r2 score of the data
        slope : slope of the regression line
        alpha : alpha for the confidence band calculation
    """

    ##### plot histogram in upper subplot
    plt.sca(ax0)

    ##### add title
    plt.title("Protein: "+str(protein_id)+"\n RANSAC Linear Regression of Sample " + str(sample))

    ##### plot histogram
    histplot(df_train, x="Reference intensity", bins=150, 
             edgecolor="lightblue", color="lightblue")

    ##### remove axis and tick labels
    plt.xlabel('')
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    ##### plot scatter plot
    plt.sca(ax1)
    
    ##### create peptide legend if sample has 20 peptides or less otherwise do not create a legend
    if len(df_train) <= 20:
        scatterplot(x="Reference intensity", y="Sample intensity", 
                    data=df_train, hue=df_train.index)
    else:
        scatterplot(x="Reference intensity", y="Sample intensity", 
                    data=df_train, hue=df_train.index, legend=False)

    ##### draw regression line
    line_label = "R2 model: " + str(r2_model) + "\nR2 data: " + str(r2_data)
    max_int = df_train["Reference intensity"].max()
    X = [0, max_int]
    y = [0, slope * max_int]
    plt.plot(X, y, color="darkblue", linestyle='-', label=line_label)

    ##### draw confidence band
    lineplot(x="Reference intensity", y="CB low", data=df_train, 
             color="darkgreen", label="CB, alpha=" + str(alpha))
    lineplot(x="Reference intensity", y="CB high", data=df_train, 
             color="darkgreen")

    ##### set line style of CB lines to dashed
    for i in [1, 2]:
        ax1.lines[i].set_linestyle("--")

    ##### show peptide legend if sample has 20 peptides or less otherwise show only regression legend
    if len(df_train) <= 20:
        ##### set right x axis limit
        plt.gca().set_xlim(right=1.4 * max_int)
        plt.legend(loc="lower right")
    else:
        plt.legend(loc="lower right")

    ##### set axis labels
    plt.ylabel("Intensity sample " + str(sample))
    plt.xlabel("Reference intensity")



def calc_raw_scores(df_distance, median_intens):
    """
    Calculates the raw scores
    Parameters:
         df_distance : pandas dataframe containing the vertical distances to the regression line
                        rows: samples, columns: peptides
         median_intens : pandas series containing the median intensities for each 
                        peptide of the reference samples
    Returns:
        pandas dataframe of same dimension as df_distance containing raw scores
    """
    ##### copy df_distance
    df_rs = df_distance.copy()

    ##### iterate through rows of df_distance (samples)
    for idx, row in df_distance.iterrows():
        ##### extract slope
        slope = row["Slope"]

        ##### delete slope from row
        row.drop("Slope", inplace=True)

        ##### calculate raw scores
        raw_scores = 1 - row / (slope * median_intens)

        ##### add slope to raw scores
        raw_scores["Slope"] = slope

        ##### replace idx row in df_RM_score with calculated raw scores
        df_rs.loc[idx] = raw_scores

    return df_rs



def normalize_t3median(df):
    """
    Applies Top3 median normalization to df
    Determines the median of the three highest values in each row and divides every value in the row by it

    Parameter:
        df : pandas dataframe of datatype float or integer

    Returns:
        normalized pandas dataframe of same dimensions as input df
    """
    ##### copy df
    df_t3med = df.copy()

    ##### for each row, normalize values by dividing each value by the median
    ##### of the three highest values of the row
    ##### iterate over rows of df
    for idx, row in df.iterrows():
        ##### calculate the median of the three highest values
        median_top3 = row.nlargest(3).median()
        
        ##### normalize each value of row by dividing by median_top3
        row_norm = row / median_top3

        ##### update row in df_norm with row_norm
        df_t3med.loc[idx] = row_norm
    
    return df_t3med



def run_FQLF(df_intens_matrix, protein_id, ref_str, num_ransac_init, mod_cutoff, 
             remove_outlier_peptides, input_file_name, output_folder, 
             reg_plots_pdf_file, scatter_plots_pdf_file):
    """
    Adapted FLEXIQuant-LF computation for multiFLEX-LF:
    Runs the RANSAC on the given matrix of peptide intensities of one protein 
    and prints dataframe of the raw scores, RM scores, modifications and 
    removed peptides for the protein into the respective CSV files.
    A dataframe of the calculated RM scores is returned.

    Parameters:
        df_intens_matrix : pandas dataframe of the intensities of the protein
                            rows: groups and samples, columns: proteins and peptides
        protein_id : str of the ID of the protein
        ref_str : str of the reference group
        num_ransac_init : int of the number of intiations of RANSAC
        mod_cutoff : float of the modification cutoff
        remove_outlier_peptides : remove peptides with outlier raw score or not
        input_file_name : str of name of the input file
        output_folder : path to the output file
        reg_plots_pdf_file : pdf file to print the regession plots into
        scatter_plots_pdf_file : pdf file to print the scatter plots into

    Returns:
        df_RM_scores : pandas dataframe of the RM scores
    """
    
    ##### filter dataframe for reference sample group
    df_intens_matrix_reference = df_intens_matrix[df_intens_matrix.index.get_level_values("Group").astype(str) == ref_str].copy()
    
    ##### remove peptides without any values in reference group
    removed_peptides = Series(df_intens_matrix.loc[:,df_intens_matrix_reference.isna().all(axis=0) == True].columns)
    df_intens_matrix_reference.drop(removed_peptides, inplace=True, axis=1)
    df_intens_matrix.drop(removed_peptides, inplace=True, axis=1)

    ##### calculate median intensities for unmodified peptides of reference samples
    median_intens_reference = df_intens_matrix_reference.median(axis=0)
    
    ##### delete columns where all entries are nan
    df_intens_matrix.dropna(how="all", axis=1)
    
    
    ##### skip protein if for none of the samples have at least 5 peptides with intensities
    if df_intens_matrix.drop(df_intens_matrix.loc[df_intens_matrix.count(axis=1) < 5].index, axis=0).empty:
        removed_peptides = removed_peptides.append(Series(df_intens_matrix.columns))
        removed_peptides = removed_peptides.sort_values().reset_index(drop=True)        
        removed_peptides = DataFrame(removed_peptides)
        removed_peptides["ProteinID"] = protein_id
        removed_peptides = removed_peptides.set_index(["ProteinID"])
        
        ##### save removed peptides as csv file, raise permission error if file can not be accessed
        path_out = add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_removed_peptides.csv")
        removed_peptides.to_csv(path_out, sep=',', mode='a', index=True, header=False)
        return
    
    ##### initiate empty lists for the results of the best model of the linear regressions
    list_slope_best = []
    list_r2_model_best = []
    list_r2_data_best = []
    list_reproducibility = []
    
    ##### copy df_intens_matrix
    matrix_distance_reg_line = df_intens_matrix.copy()
    
    ##### initiate figure for regression plots
    if reg_plots_pdf_file != '':        
        fig_protein = plt.figure(figsize=(16*2, 9*int(df_intens_matrix.shape[0]/2)+2))
        gs0 = fig_protein.add_gridspec(int(df_intens_matrix.shape[0]/2)+1, 2)
    
    plot_i = 0 # counter for the current regression plot
    
    ##### iterate through rows (samples) of df_intens_matrix
    for idx, row in df_intens_matrix.iterrows():
        
        ##### create dataframe with sample intensities and median intensities
        df_train = DataFrame({"Sample intensity": row, "Reference intensity": median_intens_reference})
        
        ##### remove nan values
        df_train.dropna(inplace=True)
        
        ##### sort dataframe
        df_train.sort_index(inplace=True, axis=0)

        ##### if number of peptides is smaller than 5, skip sample and continue with next iteration
        if len(df_train["Sample intensity"]) < 5:
            ##### set all metrics to nan
            matrix_distance_reg_line.loc[idx] = nan
            df_train["CB low"] = nan
            df_train["CB high"] = nan
            list_slope_best.append(nan)
            list_r2_model_best.append(nan)
            list_r2_data_best.append(nan)
            list_reproducibility.append(nan)
            ##### continue with next iteration
            continue
        
        ##### initiate empty list to for intermediate results of the linear regression iterations
        list_model_slopes = []
        list_r2_scores_model = []
        list_r2_scores_data = []

        ##### set training data
        X = array(df_train["Reference intensity"]).reshape(-1, 1)
        y = df_train["Sample intensity"]
        
        ##### calculate squared MAD
        sq_mad = square(df_train["Sample intensity"].mad())
        
        ##### run ransac linear regression num_ransac_init times to select best fitting model
        for i in range(num_ransac_init):
            
            ##### initiate linear regression model with ransac regressor
            ransac_model = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=False, n_jobs=-2),
                                                        max_trials=1000,
                                                        stop_probability=1,
                                                        min_samples=0.5,
                                                        loss="squared_loss",
                                                        residual_threshold=sq_mad)
            
            
            ##### fit the ransac model
            ##### skip the model if RANSAC can not fit more than 2 inlier or outlier
            with warnings.catch_warnings():
                ##### catch the warning and skip the model if it is thrown
                warnings.filterwarnings("error", category=UndefinedMetricWarning)
                
                ##### fit the model, skip if model can not be fitted and save nan values for iteration
                try: ransac_model.fit(X, y)
                except:
                    list_model_slopes.append(nan)
                    list_r2_scores_model.append(nan)
                    list_r2_scores_data.append(nan)
                    continue
            
            ##### get coefficient
            slope = (float(ransac_model.estimator_.coef_))
            
            ##### get inlier and outlier
            inlier_mask = ransac_model.inlier_mask_

            ##### add outlier as column to df_train
            df_train["Outlier"] = ~inlier_mask.astype(bool)
            
            ##### calculate r2 score based on inliers
            df_train_inlier = df_train[df_train["Outlier"] == False]
            X_inlier = array(df_train_inlier["Reference intensity"]).reshape(-1, 1)
            y_inlier = df_train_inlier["Sample intensity"]
            r2_score_model = round(ransac_model.score(X_inlier, y_inlier), 5)
            r2_score_data = round(ransac_model.score(X, y), 5)

            ##### save model and r2 score to corresponding lists
            list_model_slopes.append(slope)
            list_r2_scores_model.append(r2_score_model)
            list_r2_scores_data.append(r2_score_data)

        ##### determine best model based on r2 scores
        
        ##### if no models were found skip the sample
        if isnan(list_r2_scores_model).all():
            matrix_distance_reg_line.loc[idx] = nan
            df_train["CB low"] = nan
            df_train["CB high"] = nan
            list_slope_best.append(nan)
            list_r2_model_best.append(nan)
            list_r2_data_best.append(nan)
            list_reproducibility.append(nan)
            continue
        
        ##### remove NaNs before finding the best model, 
        ##### the max-function can not handle NaNs
        set_r2_model = set(list_r2_scores_model)
        try:
            set_r2_model.remove(nan)
        except:
            pass
        
        list_r2_scores_model_new = list(set_r2_model)
        
        ##### find index of the best model
        best_model = list_r2_scores_model.index(max(list_r2_scores_model_new))
        
        ##### save slope of best model to list_slope_best
        list_slope_best.append(list_model_slopes[best_model])
        slope = list_model_slopes[best_model]

        ##### calculate reproducibility factor and save to list
        series_slopes = Series(list_model_slopes)
        reproducibility_factor = max(series_slopes.value_counts()) / num_ransac_init
        list_reproducibility.append(reproducibility_factor)
        
        ##### get r2 scores of best model
        r2_score_model = list_r2_scores_model[best_model]
        r2_score_data = list_r2_scores_data[best_model]

        ##### save best r2 score to lists
        list_r2_model_best.append(r2_score_model)
        list_r2_data_best.append(r2_score_data)
        
        ##### calculate predicted intensities
        pred_ints = median_intens_reference * slope
        
        ##### calculate distance to regression line
        distance_reg_line = pred_ints - row
        
        ##### save distances in matrix_distance
        matrix_distance_reg_line.loc[idx] = distance_reg_line

        ##### check if plots need to be created
        if reg_plots_pdf_file != '':
            ##### calculate confidence band
            alpha = 0.3
            df_train = calculate_confidence_band(slope, median_intens_reference,
                                                 df_train, X, y, alpha)
                        
            ##### plot scatter plot with regression line into current subfigure          
            gs1 = gs0[plot_i].subgridspec(2, 1, height_ratios=[1, 6])
            ##### create axes for the regression plot subfigures
            ax1 = fig_protein.add_subplot(gs1[1])
            ax0 = fig_protein.add_subplot(gs1[0], sharex=ax1)
        
            ##### set space between subplots
            create_regression_plots(ax0, ax1, df_train, idx[1], protein_id, 
                                    r2_score_model, r2_score_data, slope, alpha)

        plot_i+=1
    
    ##### save plot to pdf file
    if reg_plots_pdf_file != '':
        reg_plots_pdf_file.savefig(figure=fig_protein, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        pass
    
    ##### add slope to dataframe
    matrix_distance_reg_line["Slope"] = list_slope_best
    
    ##### calculate raw scores
    df_raw_scores = calc_raw_scores(matrix_distance_reg_line, median_intens_reference)

    ##### calculate MAD per sample
    df_raw_scores.drop("Slope", axis=1, inplace=True)
    df_raw_scores_T = df_raw_scores.T
    
    ##### remove peptides which do not have any raw scores
    to_remove = Series(df_raw_scores_T[df_raw_scores_T.isna().all(axis=1)].index)
    df_raw_scores_T.drop(to_remove, inplace=True, axis=0)
    removed_peptides = removed_peptides.append(to_remove)
    to_remove = Series(dtype="object")
    
    ##### calculate MAD per sample
    mad = median_abs_deviation(df_raw_scores_T, scale=1, axis=0, nan_policy="omit")
    
    median = df_raw_scores_T.median(axis=0)

    ##### calculate cutoff value for each time point (> 3*MAD)
    cutoff = median + 3 * mad
    
    ##### remove raw scores with raw scores > cutoff for each sample
    df_raw_scores_T_cutoff = df_raw_scores_T[round(df_raw_scores_T, 5) <= round(cutoff, 5)]

    ##### if selected remove peptide over the outlier cutoff score
    if remove_outlier_peptides:
        to_remove = Series(df_raw_scores_T[round(df_raw_scores_T, 5) > round(cutoff, 5)].dropna(how="all", axis=0).index)        
      
    ##### add peptides without any raw scores
    to_remove = to_remove.append(Series(df_raw_scores_T_cutoff[df_raw_scores_T_cutoff.isna().all(axis=1)].index))
    
    ##### remove the peptides from the dataframe
    df_raw_scores_T_cutoff.drop(to_remove, inplace=True, axis=0)
    
    removed_peptides = removed_peptides.append(to_remove)
    removed_peptides = removed_peptides.sort_values()
 
    df_raw_scores_cutoff = df_raw_scores_T_cutoff.T

    ##### apply t3median normalization to calculate RM scores
    df_RM_scores = normalize_t3median(df_raw_scores_cutoff)
    
    ##### check if peptides are modified (RM score below modification cutoff)
    df_RM_scores_mod = df_RM_scores < mod_cutoff

    ##### add metrics of regline to df_raw_scores
    df_raw_scores["Slope"] = list_slope_best
    df_raw_scores["R2 model"] = list_r2_model_best
    df_raw_scores["R2 data"] = list_r2_data_best
    df_raw_scores["Reproducibility factor"] = list_reproducibility
    
    ##### create scatter plots of the peptide intensity and the RM scores
    if scatter_plots_pdf_file != '':
        ##### reduce dimension of matrix to a list of intensities
        df_intens_matrix_stacked = df_intens_matrix.stack()[df_RM_scores.stack().index]
        ##### create scatter plots of the intensities and the RM scores
        fig = plt.figure(figsize = (7, 7))
        scatterplot(x=log10(df_intens_matrix_stacked), y=df_RM_scores.stack(), color="lightblue")
        plt.title("Protein: "+str(protein_id))
        plt.ylabel("RM score")
        plt.xlabel("Log10 of Intensity")
        ##### save figure to pdf file
        scatter_plots_pdf_file.savefig(figure=fig, bbox_inches="tight", dpi=300)
        plt.close()

    ##### add the protein ids to the dataframes and set the index to protein and peptide ids
    df_raw_scores = df_raw_scores.T
    df_raw_scores["ProteinID"] = protein_id
    df_raw_scores = df_raw_scores.reset_index().set_index(["ProteinID", "PeptideID"])
    
    df_RM_scores = df_RM_scores.T
    df_RM_scores["ProteinID"] = protein_id
    df_RM_scores = df_RM_scores.reset_index().set_index(["ProteinID", "PeptideID"])
    
    df_RM_scores_mod = df_RM_scores_mod.T
    df_RM_scores_mod["ProteinID"] = protein_id
    df_RM_scores_mod = df_RM_scores_mod.reset_index().set_index(["ProteinID", "PeptideID"])
    
    removed_peptides = DataFrame(removed_peptides)
    removed_peptides["ProteinID"] = protein_id
    removed_peptides = removed_peptides.set_index(["ProteinID"])
    
    ##### save raw scores as csv file, raise permission error if file can not be accessed
    path_out = add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_raw_scores.csv")
    try: df_raw_scores.to_csv(path_out, sep=',', mode='a', index=True, header=False, float_format="%.5f")
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()
    
    ##### save RM scores as csv file, raise permission error if file can not be accessed
    path_out = add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_RM_scores.csv")
    try: df_RM_scores.to_csv(path_out, sep=',', mode='a', index=True, header=False, float_format="%.5f")
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()
    
    ##### save differentially modified dataframe as csv file, raise permission error if file can not be accessed
    path_out = add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_diff_modified.csv")
    try: df_RM_scores_mod.to_csv(path_out, sep=',', mode='a', index=True, header=False, float_format="%.5f")
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()
    
    ##### save removed peptides in ...mFQ-LF-output_removed_peptides.csv file, raise permission error if file can not be accessed
    path_out = add_filename_to_path(output_folder, input_file_name, "mFQ-LF-output_removed_peptides.csv")
    try: removed_peptides.to_csv(path_out, sep=',', mode='a', index=True, header=False, float_format="%.5f")
    except PermissionError:
        ##### print error message and terminate the process
        print("ERROR: Permission denied!\n" + "Please close " + path_out)
        exit()
    
    return round(df_RM_scores, 5)



def create_RM_score_distribution_plots(df_RM_scores, distri_plots_pdf_file, list_groups):
    """
    Constructs a figure of distribution plots of the RM scores. For every group a seperate plot is created 
    with the different samples in different colors.

    Parameters_
        df_RM_scores : dataframe of the RM scores of all proteins
        distri_plots_pdf_file : pdf file for printing the plots
        list_groups : list of the sample groups

    """
    
    ##### initialize the figure with a size that accounts for 7ptx7pt plots of every sample group
    fig_size = sqrt(len(list_groups))
    
    if fig_size % 1 != 0:
        fig_size=int(int(fig_size)+1)
    else:
        fig_size=int(fig_size)
    
    fig = plt.figure(figsize=(7*fig_size,7*fig_size))
    
    ##### list of colors for the color coding of the different samples in one group
    colors_list = color_palette("husl", Counter(df_RM_scores.columns.get_level_values("Group")).most_common(1)[0][1]) #int(df_group_all_prots.shape[1]/2)
    
    ##### create the distribution plots for every group and apply kernel density estimation if possible
    i=1
    for group in list_groups:
        df_group = df_RM_scores[group]
        
        ax = fig.add_subplot(fig_size,fig_size,i)
        
        try: histplot(df_group, ax=ax, kde=True, stat="count", palette=colors_list[:df_group.shape[1]], edgecolor=None)
        except: histplot(df_group, ax=ax, kde=False, stat="count", palette=colors_list[:df_group.shape[1]], edgecolor=None)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlim(0, 2)
        plt.title("Group: "+group, fontsize=16)
        plt.xlabel("RM score", fontsize=12)
    
        plt.tight_layout(h_pad=2)
        
        i+=1
    
    fig.suptitle("Distribution of RM scores of FLEXIQuant-LF ", fontsize=20)
    plt.subplots_adjust(top=0.90)
    
    ##### save figure to the pdf file
    distri_plots_pdf_file.savefig(figure=fig, bbox_inches="tight", dpi=300)
    plt.close()  
    distri_plots_pdf_file.close() 



def create_heatmap(df_RM_scores, protein_id, heatmaps_pdf_file, list_groups, cmap, norm):
    """ 
    Constructs a heatmap of the RM scores for the protein "protein_id"
    
    Parameters:
        df_RM_scores : pandas dataframe of the RM scores
        protein_id : string of the protein id
        heatmaps_pdf_file : pdf file to save the heatmap in
        list_groups : list of the sample groups
        cmap, norm : colormap and norm for the heatmap
    """
    
    ##### number of samples given
    num_samples = len(df_RM_scores.columns)
    
    ##### create matplotlib figure
    fig = plt.figure(figsize = (num_samples/3, len(df_RM_scores)/9))
    
    
    i=0 ##### counter for the position of the group-wise subplots in the figure
    
    ##### for every group add subplot of the heatmap to the figure
    for group in list_groups:
        
        ##### dataframe of the current group
        df_group = df_RM_scores.loc[:, group]
        
        ##### create axis for subplot with size = number of samples in group 
        ax = plt.subplot2grid((1, num_samples+1), (0, i), colspan=len(df_group.columns))
        
        ##### if group the first group add y-axis labels otherwise do not
        if i == 0:
            ##### create heatmap with x- and y-axis labels
            hmap = heatmap(df_group, ax=ax, cmap=cmap, norm=norm, cbar=False, xticklabels=1, yticklabels=1, annot=True, annot_kws={"size":3, "ha":"center"})
            
            ##### set the title of y-axis and the title of the figure
            plt.xlabel('')
            plt.ylabel("Peptides", fontsize=9)
            plt.title("Protein: "+protein_id, ha="center", fontsize=11)
        else:
            ##### create heatmap with only x-axis labels
            hmap = heatmap(df_group, ax=ax, cmap=cmap, norm=norm, cbar=False, xticklabels=1, yticklabels=False, annot=True, annot_kws={"size":3, "ha":"center"})
            plt.xlabel('')
            plt.ylabel('')
        
        ##### rotate and set textsize for x- and y-axis labels
        plt.setp(hmap.get_xticklabels(), rotation=90, fontsize=6)
        plt.setp(hmap.get_yticklabels(), rotation=0, fontsize=6)
        
        ##### adjust space between subplots
        plt.subplots_adjust(wspace = .2)
        
        i+=len(df_group.columns)
    
    ##### add colorbar to figure
    ax = plt.subplot2grid((1, num_samples+1), (0, i), colspan=1)
    cb = plt.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=ax, orientation="vertical", extend="max")
    cb.ax.tick_params(labelsize=6)
    cb.ax.set_ylabel("RM score", fontsize=8)
    cb.outline.set_visible(False) 
    
    ##### save figure in pdf file and close it
    heatmaps_pdf_file.savefig(figure=fig, bbox_inches="tight", dpi=300)
    plt.close()

    
    
    
def RM_score_distance(u, v, mod_cutoff):
    """
    Calculation of the customized Manhattan distance between the arrays of RM scores u and v 

    Parameters
        u : array RM scores of a peptide
        v : array RM scores of a
        mod_cutoff : user defined modification cutoff

    Returns:
        RM score distance of u and v
    """
    ##### initialize distance vector
    if len(u) == len(v): dist = ones(len(u))
    else: return 
    
    ##### calculate absolute differences between the elements of u and v
    for i in range(len(dist)):
        x = u[i]
        y = v[i]
        
        ##### penalize jumps from below to above the modification cutoff
        if (x < mod_cutoff and y >= mod_cutoff) or (x >= mod_cutoff and y < mod_cutoff):
            dist[i] = abs(x-y)+1
        else:
            dist[i] = abs(x-y)
    
    ##### return sum of the penalized absolute differences
    return dist.sum()


def missing_value_imputation(df_RM_scores, max_cos_dist):
    """
    Impute missing values by calculating the median of the RM scores of all peptides 
    with a cosine distance (i.e. 1 - cosine similarity) of at most max_cos_dist from
    the current peptide. If not all missing values of a peptide were imputed, it is 
    removed from further analysis.
    
    Parameters:
        df_RM_scores : pandas dataframe of RM scores
        max_cos_dist : integer between 0 and 1

    Returns:
        df_RM_scores : pandas dataframe of RM scores without removed peptides
        df_RM_scores_imputed : pandas dataframe of RM scores with the imputed values
                               without removed peptides
        pd.DataFrame(list(remove_nans)) : dataframe of the removed peptides
    """

    ##### copy df
    df_RM_scores_imputed = df_RM_scores.copy()
    
    for peptide in df_RM_scores.index:

        ##### get RM scores of the current peptide
        df_RM_scores_pep = df_RM_scores.loc[peptide]
        
        ##### skip if no missing value for peptide
        if not df_RM_scores_pep.isna().any(): continue
        
        ##### remove NaN values
        df_RM_scores_pep = DataFrame(df_RM_scores_pep.dropna())

        ##### get all other peptides and keep only samples that have a RM scores for the current peptides
        df_RM_scores_other_peps = df_RM_scores[df_RM_scores_pep.index].drop(peptide, axis=0)
        
        ##### calculate all pairwise cosine distances between the current peptide and all other
        cos_dist_other_peps = cdist(df_RM_scores_pep.T, df_RM_scores_other_peps, "cosine")[0]
        
        ##### get the index of the closest peptides
        index_impute = df_RM_scores_other_peps[cos_dist_other_peps <= max_cos_dist].index
        
        ##### skip peptide if less than 2 close peptides were found
        if len(index_impute) < 2: continue
        
        ##### calculate the median RM scores of the closest peptides
        df_imputation_values = df_RM_scores.loc[index_impute].median().drop(df_RM_scores_pep.index)
        
        ##### replace the missing values with the calculated values
        df_RM_scores_imputed.loc[peptide, df_imputation_values.index] = df_imputation_values

    ##### remove all peptides that still have missing values
    remove_nans = df_RM_scores_imputed[df_RM_scores_imputed.isna().any(axis=1)].index
    df_RM_scores_imputed = df_RM_scores_imputed.drop(remove_nans, axis=0)
    df_RM_scores = df_RM_scores.drop(remove_nans, axis=0)
    
    return df_RM_scores, df_RM_scores_imputed, DataFrame(list(remove_nans))


def peptide_clustering(df_RM_scores, linkage_matrix, mod_cutoff, cmap, colors, clust_threshold, clust_ids):
    """
    Clustering results are saved as interactive HTML file with the dendrogram and the heatmap.

    Parameters:
        df_RM_scores : dataframe of RM scores
        linkage_matrix : linkage matrix of the hierarchical clustering
        mod_cutoff : user defined cutoff
        cmap : name of the pyplot color map to use
        colors: list of colors used in the dendrogram with the threshold
        clust_threshold: distance threshold of clusters for dendrogram
        clust_ids: list of the cluster ids of the peptides and proteins based on the distance theshold
    """
    
    ##### initialize plotly figure and create the dendrogram based on the linkage matrix
    plotly_figure = create_dendrogram(df_RM_scores, 
                               orientation="right",
                               linkagefun=lambda x: linkage_matrix,
                               colorscale=colors,
                               color_threshold=clust_threshold,
                               )

    ##### set x-axis of the dendrogram to x2 (axis nr. 2)
    for i in range(len(plotly_figure["data"])): plotly_figure["data"][i]["xaxis"] = "x2"
    
    ##### get order of the peptides in the dendrogram
    clust_leaves = plotly_figure["layout"]["yaxis"]["ticktext"]
    clust_leaves = list(map(int, clust_leaves))
    
    ##### create numpy array from the RM scores dataframe 
    ##### and sort the peptides by the order in the dendrogram
    heat_data = df_RM_scores.to_numpy()
    heat_data = heat_data[clust_leaves,:]
    
    ##### define row and column labels for the heatmap
    row_names = [str(i[0])+"<br />Peptide: "+str(i[1]) for i in df_RM_scores.index]
    row_names = list(array(row_names)[clust_leaves])
    col_names = list(df_RM_scores.columns.get_level_values(1))
    
    if len(clust_ids) == 0:
        ##### define the row IDs shown upon hovering over the cells of the heatmap
        clust_id = array([[i]*df_RM_scores.shape[1] for i in range(len(clust_leaves)-1,-1,-1)])
    else:
        ###### if cluster ids given add the cluster id to the hover information
        clust_id = array([[str(i)+"<br />Cluster: "+str(clust_ids[i])]*df_RM_scores.shape[1] for i in range(len(clust_leaves)-1,-1,-1)])
        
    
    ##### create heatmap
    heatmap = Heatmap(z = heat_data,
                         colorscale = cmap,
                         zmin = 0, zmax = mod_cutoff*2,
                         customdata = clust_id,
                         hovertemplate = "Sample: %{x}<br />Protein: %{y}<br />RM score: %{z}<br />ID: %{customdata}",
                         colorbar=dict(title="RM score", 
                                       titleside="top", 
                                       tickmode="array",
                                       thicknessmode="pixels", 
                                       thickness=25, 
                                       lenmode="pixels", 
                                       len=250, 
                                       yanchor="top", 
                                       y=1, x=1.05, 
                                       ticks="outside",
                                       dtick=5))
    
    ##### align y-axis of heatmap to dendrogram
    heatmap["y"] = plotly_figure["layout"]["yaxis"]["tickvals"]
        
    #### add the heatmap to plotly figure plotly_figure
    plotly_figure.add_trace(heatmap)
    
    
    ##### edit layout of plotly_figure
    figure_width = min(max(df_RM_scores.shape[1]*50+100, 500), 1500)
    plotly_figure.update_layout(width = figure_width,
                      height = 900, font={"size": 11})
    
    ##### amount of the figure size used for the dendrogram
    dendrogram_width = 100/figure_width
    
    ##### update x-axis of the heatmap
    plotly_figure.update_layout(xaxis={"domain": [dendrogram_width, 1],
                             "mirror": False,
                             "showgrid": False,
                             "showline": False,
                             "zeroline": False,
                             "showticklabels": True,
                             "ticks": "outside",
                             "ticktext": col_names, # list of sample names
                             "tickvals": arange(0, len(col_names))
                     })
    
    ##### update x-axis (xaxis2) of the dendrogram
    plotly_figure.update_layout(xaxis2={"domain": [0, dendrogram_width],
                              "mirror": False,
                              "showgrid": False,
                              "showline": False,
                              "zeroline": False,
                              "showticklabels": True,
                              "ticks":''
                      })
    
    ##### update y-axis of the heatmap
    plotly_figure.update_layout(yaxis={"domain": [0, 1],
                             "mirror": False,
                             "showgrid": False,
                             "showline": False,
                             "zeroline": False,
                             "showticklabels": False,
                             "ticks": '',
                             "side": "right",
                             "ticktext": row_names # list of the peptides
                    })
    
    ##### set plot title and axes lables
    plotly_figure.update_layout(title="Clustered Heatmap of RM scores",
                      xaxis_title="Sample",
                      yaxis_title="Peptides",
                      xaxis2_title="",
                      template="plotly_white"
                      )
    
    ##### return the figure, the matrix of the RM scores in clustering order and the clustering order of the peptides
    return plotly_figure, heat_data, clust_leaves
