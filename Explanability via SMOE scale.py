import numpy as np
import numpy.random as npr
from datetime import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import boxcar, welch, hamming, savgol_filter, find_peaks, filtfilt, butter
from scipy import stats
from scipy.stats import pearsonr, mannwhitneyu, sem, skew, kurtosis, describe, kstest, spearmanr, shapiro, chisquare, ks_2samp
from scipy.optimize import minimize
from scipy.special import logit, softmax, expit, erf, rel_entr
import sys
import seaborn as sns
import tsfresh.feature_extraction.feature_calculators as tsfeat
from datetime import datetime as dt
from statsmodels.stats.weightstats import DescrStatsW
import itertools

### sklearn
from sklearn.cluster import KMeans, SpectralClustering, FeatureAgglomeration, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, roc_curve 
from sklearn.metrics import silhouette_samples, matthews_corrcoef, r2_score, explained_variance_score
from sklearn.metrics.pairwise import euclidean_distances, paired_distances, cosine_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Binarizer, LabelEncoder, PowerTransformer
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA, KernelPCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression, Perceptron, LogisticRegressionCV
from sklearn.feature_selection import mutual_info_classif, RFECV, RFE
from tslearn import shapelets
from tslearn.clustering import KShape, silhouette_score, TimeSeriesKMeans
from tslearn.piecewise import PiecewiseAggregateApproximation as PAA
from tslearn.preprocessing import TimeSeriesScalerMeanVariance 

### ###
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

### ### ###
import tensorflow.keras.backend as K
import tensorflow as tf

### ### ###
import ICUAI_OCM as ocm #(for more details see  https://github.com/glancia93/ICUAI-dynamic-prediction/blob/main/ICUAI_module.py)

###
import SMOE #(for more details see https://github.com/glancia93/Physics-captured-by-data-based-methods-in-El-Nino-prediction_PyCODE/blob/main/SMOE.py )

#### #### #### #### ####
def find_best_location(X, model_cnn, time_scale_smoe = 8):
    
    ###compute smoes (Xtest)
    smoe_masks = []
    hiddenmaps = SMOE.get_feat_maps(X, model_cnn)
    magic_numbers = [1, 5, 9]
    for npat in range(0, X.shape[0]):
        list_actv_feat_maps = [np.expand_dims(hiddenmaps[jj][npat], axis= 0) for jj in magic_numbers]
        smoe_mask = SMOE.SMOEcombined(list_actv_feat_maps, lenght_smoe=X.shape[1], weights='exp').ravel()
        smoe_masks.append(smoe_mask)
    smoe_masks = np.vstack(smoe_masks)

    length_flat_mask = int(np.round((time_scale_smoe*60/9), 0))
    
    
    ##compute smoothed smoeos
    smoothed_smoes = np.apply_along_axis(np.convolve, 
                                         axis= 1, 
                                         arr= smoe_masks, 
                                         v= flat_mask(length_flat_mask), 
                                         mode= 'valid')

    ###get best smoe location (as intervals)
    best_smoe_loc = np.argmax(smoothed_smoes, axis= 1)
    best_smoe_interval = np.array([best_smoe_loc, best_smoe_loc+length_flat_mask]).T
    return best_smoe_interval

 def most_salient_extraction(X, location):
    
    
    list_feats= []
    for npat in range(X.shape[0]):
        list_of_the_pat = []
        loc0, loc1= location[npat]
        ts_internal = X[npat, loc0:loc1, :]
        list_feats.append(ts_internal)

    return np.array(list_feats)

def flat_mask(Npoints):
    return np.ones(Npoints)/Npoints

def data_driven_clustering(X, cls= 1, mark= 'NONE'):
    
    data_ = X[:, :, [0, 1, 3, 4]].mean(axis= 1)
    data_bin= data_.copy()
    data_bin[:, 0] = (data_bin[:, 0]>=90).astype(int) # HR
    data_bin[:, 1] = (data_bin[:, 1]<=65).astype(int) # ABPm
    data_bin[:, 2] = (data_bin[:, 2]<=95).astype(int) # SaO2
    data_bin[:, 3] = (data_bin[:, 3]>=24).astype(int) # BR
    
    ### describe as classes. from binary to classes
    data_class = np.zeros(data_bin.shape[0])
    for kk in range(data_bin.shape[0]):
        data_class[kk] = (np.power(2, np.arange(0, data_bin.shape[1], 1))*data_bin[kk]).sum()
    
    ###to float
    freq_, bins_ = np.histogram(data_class, bins=np.arange(0, 16, 1), density=False)
    print(chisquare(f_obs=freq_))
    kolors= ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']
    plt.hist(data_class, bins=np.arange(0, 16, 1), color=kolors[cls], density=True)
    plt.grid(True)
    plt.ylabel('Frequencies')
    plt.xlabel('Classes')
    plt.xticks(.5+np.arange(0, 16, 1), np.arange(0, 16, 1))
    plt.savefig('Hist_datadrivenclue_lm'+str(mark)+'_cls'+str(cls)+'.png')
    plt.show()
    return data_class
  
  
  def univariate_class_correlation(X, Y):
    
    """
    X--> data class 0
    Y--> data class 1
    """
    
    ### determine range where we seach for the best threshold
    min_= np.minimum(X.min(), Y.min())
    max_ = np.maximum(X.max(), Y.max())
    range_ = np.linspace(min_, max_, 1000)
    
    ###
    U, V = np.hstack((X, Y)), np.hstack((np.zeros(X.shape[0]), np.ones(Y.shape[0])))
    coeffs= []
    for th in range_:
        W = (U>=th).astype(int)
        coeffs.append(matthews_corrcoef(y_pred=W, y_true= V))
    
    coeffs= np.array(coeffs)
    wheremax_= np.argmax(coeffs)                                    
    print('%%%%%%%%%%%%%')
    print('Best Threshold:', range_[wheremax_].round(1))
    print('Corr. Coef. :', coeffs[wheremax_].round(2))
    
    return 
