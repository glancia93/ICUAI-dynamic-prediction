###
import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau,LearningRateScheduler

###
import pywt
from datetime import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import seaborn as sns
import numpy.fft as npf
import pandas as pd
import tsfresh.feature_extraction.feature_calculators as tsfeat
from scipy.stats import iqr

##################
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity, rbf_kernel

###
from scipy.signal import argrelextrema, hamming, tukey, gaussian, butter, filtfilt, boxcar
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis, entropy

###################################################################

def remove_mean(X):
    for jj in range(0, X.shape[0]):
        X[jj] = X[jj]-X[jj].mean()
    return X

def meanpooling(X, pooling):
        ## determina il numero di chunks
        N = int(X.size/pooling)
        ## 
        Y = X.copy()[0:N*pooling].reshape(N, -1)
        return Y.mean(axis= 1)
    
#############################################
def reduce_scale(data, cov, zero_one= False):
    data_copy = data.copy()
    
    if cov == 'NONE':
        return data_copy
    
    ###
    # case [0, 1]
    elif zero_one== True:
        
        minmax_dict = {'HR':[40, 150], 
                       'ABPm':[40, 120], 
                       'Pulse':[10, 120],
                       'SaO2':[80, 100],
                       'SaO2_':[-100, 0],
                       'BR1':[11, 33],
                       'etCO2':[18, 49], 
                       'RMV':[2, 17], 
                       'Temp': [28, 45], 
                        'NONE': [0, 1]}
        
        Min, Max = minmax_dict[cov]
        DELTA = Max-Min
        A, B = 1/DELTA, -Min/DELTA
        data_copy = data_copy*A+B*np.ones(data_copy.shape)
        ###
        data_copy[np.isnan(data_copy)] = .5
        data_copy[data_copy>1] = 10
        data_copy[data_copy<0] = -10
        return data_copy
   
    ## case [-1, 1]
    else: 
        minmax_dict = {'HR':[40, 150], 
                       'ABPm':[40, 120], 
                       'Pulse':[10, 120],
                       'SaO2':[80, 100],
                       'SaO2_':[-100, 0],
                       'BR1':[11, 33],
                       'etCO2':[18, 49], 
                       'RMV':[2, 17], 
                       'Temp': [28, 45], 
                        'NONE': [-1, 1]}
        
        Min, Max = minmax_dict[cov]
        DELTA = Max-Min
        A, B = 2/DELTA, -(Max+Min)/DELTA
        ### ### ###
        data_copy = data_copy*A+B*np.ones(data_copy.shape)
        ###
        data_copy[np.isnan(data_copy)] = 0
        data_copy[data_copy>1] = 1
        data_copy[data_copy<-1] = -1
        return data_copy
    
def inverse_reduce_scale(data, cov, zero_one= False):
    data_copy = data.copy()
    
    
    if zero_one:
        minmax_dict = {'HR':[40, 150], 
                       'ABPm':[40, 120], 
                       'Pulse':[10, 120], 
                       'SaO2':[80, 100],
                       'SaO2_':[-100, 0],
                       'BR1':[11, 33],
                       'etCO2':[18, 49], 
                       'RMV':[2, 17], 
                       'Temp': [28, 45], 
                       'NONE': [1, 0]}
        ###
        Min, Max = minmax_dict[cov]
        DELTA = Max-Min
        A, B = 1/DELTA, -(Max+Min)/DELTA
    else:
        minmax_dict = {'HR':[40, 150], 
                       'ABPm':[40, 120], 
                       'Pulse':[10, 120], 
                       'SaO2':[80, 100],
                       'SaO2_':[-100, 0],
                       'BR1':[11, 33],
                       'etCO2':[18, 49], 
                       'RMV':[2, 17], 
                       'Temp': [28, 45], 
                       'NONE': [-1, 1]}
        ###
        Min, Max = minmax_dict[cov]
        DELTA = Max-Min
        A, B = 2/DELTA, -(Max+Min)/DELTA
        
    data_copy = (data_copy-B)/A
    return data_copy
    
### ### ### ###
def prepare_data_CNN(Xtrain, Xtest, list_covs, pooling= 9, mask_amplitude= 60):
    
    ####
    xtrain, xtest = Xtrain.copy(), Xtest.copy()
    
    ###standardizza
    for kk in range(0, len(list_covs)):
        xtrain[:, :, kk] = reduce_scale(Xtrain[:, :, kk], list_covs[kk])
        xtest[:, :, kk] = reduce_scale(Xtest[:, :, kk], list_covs[kk])
    
    ### ### ###
    ### riduci
    xtrain_, xtest_ = xtrain.copy(), xtest.copy()
    xtrain = np.empty((Xtrain.shape[0], int(Xtrain.shape[1]/pooling), Xtrain.shape[2]))
    xtest = np.empty((Xtest.shape[0], int(Xtest.shape[1]/pooling), Xtest.shape[2]))
    for kk in range(0, xtrain.shape[2]):
        xtrain[:, :, kk] = np.apply_along_axis(meanpooling, 1, xtrain_.copy()[:, :, kk], pooling= pooling)
        xtest[:, :, kk] = np.apply_along_axis(meanpooling, 1, xtest_.copy()[:, :, kk], pooling= pooling)
    
    return xtrain, xtest

def standardize_TS(X):
    Y = X.copy()
    if np.nanstd(Y) !=0:
        return (Y-np.nanmean(Y))/np.nanstd(Y)
    else:
        return Y-np.nanmean(Y)

    
def zeroone_TS(X):
    Y = X.copy()
    Ymax, Ymin = X.max(), Y.min()
    if Ymax-Ymin !=0:
        a, b = 1/(Ymax-Ymin), -Ymin/(Ymax-Ymin)
        return a*X+b
    else:
        return Y
    
    
def zeromean_TS(X):
    Y = X.copy()
    return Y-np.mean(Y)


def meanpooling_mf(mftrain, mftest, pooling= 9):
    ### prepara la feature delle missing_flags per 
    mftrain_, mftest_ = mftrain.copy(), mftest.copy()
    mftrain_ = np.apply_along_axis(meanpooling, 1, mftrain_.copy(), pooling= pooling)
    mftest_ = np.apply_along_axis(meanpooling, 1, mftest_.copy(), pooling= pooling)
    mftrain_, mftest_ = np.expand_dims(mftrain_, axis= 2), np.expand_dims(mftest_, axis= 2)
    return mftrain_, mftest_


def extract_TS_CNN(X):
    Z = np.empty((X.shape[0],X.shape[1], 3))
    Z[:, :, 0] = np.gradient(X, axis= 1)
    Z[:, :, 1] = np.gradient(Z[:, :, 0], axis= 1)
    Z[:, :, 2] = cumtrapz(X, initial= 0, axis= 1)/X.shape[1]
    #Z[:, :, 3] = cumtrapz(Z[:, :, 0]**2, initial= 0)/X.shape[1] ##kinetic energy
    return Z
   
def mild_convolution(X, W):
    Y = np.pad(X, pad_width=(int(W.size/2), int(W.size/2)), mode= 'edge')
    return np.convolve(Y, W, mode= 'valid')[0:X.shape[0]]


def apodization(X):
    Ndim = X.shape[0]
    Y = X.copy()
    W = hamming(Ndim)
    W = W/W.sum()
    return Y*W

def vanilla_apodization(X):
    Ndim = X.shape[0]
    Y = X.copy()
    W = tukey(Ndim, alpha= 50e-2)
    return Y*W

def cosine_distribution(N= 5):
    Xn = np.linspace(0, 1, N, endpoint= True)
    cs = np.cos(2*np.pi*Xn)
    cs_prob = (cs-cs.min()+1e-8)/np.sum(cs-cs.min()+1e-8)
    return cs_prob

def exp_distribution(N= 5):
    Xn = np.arange(0, N, 1)
    cexp = np.exp(-Xn)
    cexp_prob = cexp/np.sum(cexp)
    return cexp_prob

def gaussian_distribution(N= 5):
    Cnormal = gaussian(N, std= 1)/gaussian(N, std= 1).sum()
    Cnormal_prob = Cnormal/np.sum(Cnormal)
    return Cnormal_prob

def GCU(X):
    "Growing Cosine Unit -- Activation Function"
    return K.exp(K.log(X)+K.cos(X))


def transform_mlp_data(X, list_covs, width_peaks= 31):
    feats= []
    for kk in range(0, X.shape[2]):
        print(kk, dt.now())
        feats.append(extract_mlp_feats(X[:, :, kk], list_covs[kk], width_peaks))  
        #feats.append(ocm.extract_(X[:, :, kk]))  
    
    Y = np.hstack(feats)
    return Y

def get_data_mlp(inst, mfs, list_covs):
    xx= inst.copy()
    #for kk in range(0, len(list_covs)):
    #    xx[:, :, kk] = reduce_scale(xx[:, :, kk], list_covs[kk])

    #### ESTRAI FEATS MLP
    X= transform_mlp_data(xx,  list_covs)
    #X= np.concatenate((X, mfs.mean(axis= 1).reshape(-1, 1)), axis= 1)
    ### SCALING
    Zsclaer = StandardScaler()
    Zsclaer.fit(X)
    return Zsclaer.transform(X)


def get_data_cnn(inst, labels, mfs, list_covs, pooling= False, pool_size= 9, zero_one_scale= False):
    xx= inst.copy()
    
    ### transform data
    for kk in range(0, len(list_covs)):
        xx[:, :, kk] = reduce_scale(xx[:, :, kk], list_covs[kk], zero_one= zero_one_scale)
    
    
    if pooling:
        #####
        xx_ = np.empty((xx.shape[0], int(xx.shape[1]/pool_size), xx.shape[2]))
        for kk in range(0, xx_.shape[2]):
            xx_[:, :, kk] = np.apply_along_axis(meanpooling, 1, xx.copy()[:, :, kk], pooling= pool_size)

        ### reduce missing masks
        mfs_ = np.apply_along_axis(meanpooling, axis= 1, arr= mfs, pooling= pool_size)
        mfs_ = np.expand_dims(mfs_, axis= 2)
        return np.concatenate((xx_, mfs_), axis= 2), labels
    
    else:
        return np.concatenate((xx, np.expand_dims(mfs, axis= 2)), axis= 2), labels
    
    
    ### pre-processing per le reti-2D
### funzione per trasformare i dati 1D in immagini 2D
def TS_become_Matrix(ts, dims):
    m, n  = dims # prendi le dimenzioni
    matrix = np.zeros(dims) #inizializza la matrice di uscita
    T = ts.shape[0] #lunghezza TS
    height = 1.0/m  # altezza griglia
    width = T/n  # larghezza griglia
    ###
        
    ### scegli indici -i
    ii = ((1-ts)/height).astype(int)
    cond_i = ii>=m
    ii[cond_i] = m-1
    
    ### scegli indice j
    time = 1+np.arange(0, T, 1)
    jj =  ((time)/width)
    cond_jj = 1.*jj.astype(int) == np.round(jj, 3)
    jj[cond_jj] = jj[cond_jj].astype(int)-1
    jj = jj.astype(int)
    
    #### binning
    matrix_, xbins_, ybins_ = np.histogram2d(ii, jj, bins= [m, n])
    return matrix_/matrix_.sum()

### prepara il tensore di dati 1D in dati 2D
def prepeare_data_image(X, dims):
    image_ = np.empty((X.shape[0], dims[0], dims[1], X.shape[2]))
    
    for feat in range(0, X.shape[2]):
        image_[:, :, :, feat] = np.apply_along_axis(TS_become_Matrix, axis= 1, arr= X[:, :, feat].copy(), dims= dims)
        
    return image_


def get_data_cnn2d(inst, labels, mfs, list_covs, dims= (20, 24)):
    xx= inst.copy()
    
    ### transform data
    for kk in range(0, len(list_covs)):
        xx[:, :, kk] = reduce_scale(xx[:, :, kk], list_covs[kk], zero_one= True)
    
    ### missing masks
    mfs_ = np.expand_dims(mfs, axis= 2)
    xx_ = np.concatenate((xx, mfs_), axis= 2)
    
    ### prepare data as 2D images 
    output = prepeare_data_image(xx_, dims)
    
    return output, labels
    
    
    
    ###################################
def CNN_(X_train, Y_train, 
         X_test, Y_test, 
         pooling = True, 
         init_poolsize_CNN = 9,
         filters_CNN= 8, 
         kernel_size_CNN= 5, 
         poolsize_CNN= 2,
         DropOut_CNN= 50e-2,
         strides_CNN= 1, 
         activation= 'relu', 
         stddev_pred = 1e-1, 
         padding = 'valid', 
         bias = True, 
         activation_pred= 'sigmoid', 
         deepness_CNN= 1, 
         dense_units= 1, 
         second_dense_layer = False, 
         second_dense_units = 1,
         epochs= 1000, 
         batch_size= 500, 
         verbose= 0,
         lr = 1e-3, 
         rate_decay = 1e-7, 
         winit = 1e-1,
         patience= 5, 
         print_summary= False):
    
    ###############
    Winit = tf.keras.initializers.RandomUniform(minval=-winit, maxval= winit)
    
    ### ### ### ###
    Inputs = keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
    ### ### ### ### ###
    ### 1st_layer ## ##
    
    if pooling:
        X = layers.AvgPool1D(pool_size = init_poolsize_CNN, padding= 'same')(Inputs)
        X = layers.Conv1D(filters= filters_CNN, 
                          kernel_size= kernel_size_CNN,
                          activation= 'linear', 
                          strides = strides_CNN, 
                          kernel_initializer= Winit,
                          padding = padding, 
                          use_bias = bias)(X)
        X = layers.Activation(activation)(X)
        X = layers.MaxPool1D(pool_size = poolsize_CNN, padding= 'same')(X)
        X = layers.Dropout(rate = DropOut_CNN)(X)
        
    
    else:
        X = layers.Conv1D(filters= filters_CNN, 
                          kernel_size= kernel_size_CNN,
                          activation= 'linear', 
                          strides = strides_CNN, 
                          kernel_initializer= Winit,
                          padding = padding, 
                          use_bias = bias)(Inputs)
        X = layers.Activation(activation)(X)
        X = layers.MaxPool1D(pool_size = poolsize_CNN, padding= 'same')(X)
        X = layers.Dropout(rate = DropOut_CNN)(X)
        
    for jj in range(1, deepness_CNN):
        X = layers.Conv1D(filters= filters_CNN, 
                        kernel_size= kernel_size_CNN,
                        activation= 'linear', 
                        strides = strides_CNN, 
                        padding = padding, 
                        kernel_initializer= Winit,  
                        use_bias = bias)(X)
        X = layers.Activation(activation)(X)
        X = layers.MaxPool1D(pool_size = poolsize_CNN, padding= 'same')(X)
        X = layers.Dropout(rate = DropOut_CNN)(X)

    ### flattern   
    X = layers.Flatten()(X)
    rho_rate = np.sqrt((stddev_pred**2)/(1+stddev_pred**2))
    X= layers.GaussianNoise(stddev= stddev_pred)(X)
    Xfinal = layers.Dense(units = dense_units, activation= activation_pred, use_bias= bias)(X) 
    
    ##define model
    mymodel = tf.keras.models.Model(Inputs, Xfinal)
    if print_summary:
        print(mymodel.summary())

    ### ###
    adam = keras.optimizers.Adam(lr= lr)
    bce = tf.keras.losses.BinaryCrossentropy()
    mymodel.compile(optimizer= adam,
                    loss = bce)
    
    ###################################
    ## FIT 
    #print('Network Training...')
    #########################
    ### callbacks
    
    #auc_train= IntervalEvaluation(validation_data=(X_train, Y_train), interval=5)
    #auc_test= IntervalEvaluation(validation_data=(X_test, Y_test), interval=5)
    
    ###lr_scheduler
    def LRScheduler(epoch, lr= lr, rate_decay= rate_decay):
        ### linear decreasing
        lrate = lr -rate_decay 
        return lrate
    lrate = tf.keras.callbacks.LearningRateScheduler(LRScheduler)
    
    ###early_stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                    min_delta=0, 
                                    patience= patience)
    ### call fit
    mymodel.fit(X_train, Y_train, 
              validation_data = (X_test, Y_test),
              epochs= epochs, 
              batch_size= batch_size, 
              verbose= verbose,
              callbacks=[lrate, early_stopping])
    
    
    
    return mymodel
    
    
    
    ##### CNN 2D #####
#### definisci rete neurale 2D
def CNN_2D(Xtrain, Ytrain, 
           Xtest, Ytest, 
           filters_CNN= 8, 
           kernel_size_CNN= (3, 3),
           activation = 'relu',
           strides_CNN= (1, 1), 
           poolsize_CNN= 2, 
           DropOut_CNN= 50e-2, 
           padding= 'valid',
           stddev_pred= 10e-1,
           deepness_CNN = 2, 
           lr= 1e-3, 
           early_stopping_patience= 3, 
           epochs= 1000, 
           batch_size= 500, 
           verbose= 1):
    
    ### ### ### ###
    input_shape = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])
    
    Inputs = keras.Input(shape=input_shape)
    ### ### ### ### ###
    ### 1st_layer ## ##
    X = layers.Conv2D(filters= filters_CNN, 
                          kernel_size= kernel_size_CNN,
                          activation= activation, 
                          strides = strides_CNN, 
                          padding = padding)(Inputs)
    X = layers.MaxPool2D(pool_size = poolsize_CNN, padding= 'same')(X)
    X = layers.GaussianDropout(rate = DropOut_CNN)(X)
        
    for kk in range(0, deepness_CNN-1):
        X = layers.Conv2D(filters= filters_CNN, 
                              kernel_size= kernel_size_CNN,
                              activation= activation, 
                              strides = strides_CNN, 
                              padding = padding)(X)
        X = layers.MaxPool2D(pool_size = poolsize_CNN, padding= 'same')(X)
        X = layers.GaussianDropout(rate = DropOut_CNN)(X)

    ### ### #### ### ###
    ### flattern   
    X = layers.Flatten()(X)
    X = layers.GaussianNoise(stddev= stddev_pred)(X)
    Xfinal = layers.Dense(units = 1, activation= 'sigmoid')(X) 
    ##define model
    mymodel = tf.keras.models.Model(Inputs, Xfinal)
    
    ## COMPILER
    adam = keras.optimizers.Adam(lr= lr)
    bce = tf.keras.losses.BinaryCrossentropy()
    mymodel.compile(optimizer= adam,
                    loss = bce)
    
    
    ##early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                      min_delta=0, 
                                                      patience= early_stopping_patience)
    ## FIT ##
    mymodel.fit(Xtrain, Ytrain,
              validation_data = (Xtest, Ytest),
              epochs= epochs, 
              batch_size= batch_size, 
              verbose= verbose, 
              callbacks= [early_stopping])

    
    return mymodel
