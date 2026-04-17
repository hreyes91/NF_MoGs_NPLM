import os
import re
import sys
import csv
import time
import json
import itertools
import argparse
import torch
import mplhep as hep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from datetime import datetime
from scipy import stats
from scipy import optimize
from scipy.stats import chi2, norm

from sklearn.metrics.pairwise import pairwise_distances

from reload_new import sample_nf, sample_true, sample_true_corr, load_model, load_hyperparams, create_flow, gen_sample

import time

def standardize_dataset(feature, mean_REF=[], std_REF=[]):                                                                                                                                
    feature_std = np.copy(feature)
    mean_REF_tmp=[]
    std_REF_tmp =[]
    for j in range(feature.shape[1]):
        vec = np.float32(feature_std[:, j])
        if len(mean_REF)==0:
            mean = np.mean(vec)
            mean_REF_tmp.append(mean)
        else:
            mean = mean_REF[j]
        if len(std_REF)==0:
            std  = np.std(vec)
            std_REF_tmp.append(std)            
        else:
            std  = std_REF[j]
        vec = vec - mean
        vec = vec *1./ std
        feature_std[:, j] = vec
    return np.float64(feature_std), mean_REF_tmp, std_REF_tmp


def generate_true_dataset(N_BKG=1000,ndim=4):
    
    print("GENERATING DATASET")
    
   
    
    # model_path = '/home/ubuntu/FlashSim/nplm-test-flows/NFs/Training/Mains_MoG/results/test_2_RealNVPN/run_'+ model + '/'
    # print("generate datase from", model_path)

    # reference
    true_data = sample_true(N_BKG,ndim,path_to_results='//net/data_ttk/hreyes/NPLM/nplm-gen-models//results/test_2_RealNVPN/',seed=0,ncomp=3)


    #normalize
    feature_ref, mean_REF, std_REF = standardize_dataset(true_data)
    feature_data, _, _ = standardize_dataset(true_data, mean_REF, std_REF)

    #target 
    #target_ref  = torch.zeros((N_BKG, 1), dtype=torch.float64)
    #target_data = torch.ones((N_BKG + N_S, 1), dtype=torch.float64)

    #feature = torch.cat((torch.from_numpy(feature_ref), torch.from_numpy(feature_data)), axis=0)
    #target  = torch.cat((target_ref,  target_data), axis=0)
    

    return true_data


def GenerateNFdata(N_SIG=100,iter_size=10,Model='101'):


    model_path = '//net/data_ttk/hreyes/NPLM/nplm-gen-models/results/test_2_RealNVPN/run_'+ Model + '/'
    # model_path = '/home/ubuntu/FlashSim/nplm-test-flows/NFs/Training/Mains_MoG/results/test_2_RealNVPN/run_'+ args.Model + '_new/'
    print("generate datase from", model_path)
    ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser=load_hyperparams(model_path)
    nf_arq=create_flow(bijector_name,nbijectors,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style='reverse')
    nf_dist,model=load_model(nf_arq,model_path,ndims,lr=.00001)
    
    nf_data=gen_sample(nf_dist,sample_size=N_SIG,iter_size=iter_size)

    return nf_data

def SaveData(sample,path_to_results,sample_tag):

    #print(np.shape(sample_all))
    with open(path_to_results+'/sample_'+str(sample_tag)+'.npy', 'wb') as f:
        np.save(f, sample, allow_pickle=True)
    print('samples saved')

    return

def load_sample(path_to_results,sample_tag):

    loaded_sample=np.load(path_to_results+'/sample_'+str(sample_tag)+'.npy',allow_pickle=True)
    #nf_sample=np.load(path_to_results+'/sample_nf.pcl',allow_pickle=True)
    
    
    return loaded_sample

path_to_results='/net/data_ttk/hreyes/NPLM/nplm-gen-models/results/samples'

N_BKG=1000_000
N_SIG=1000_000
#ndim=4
#Model='101'
param_dict={4:['101','103','105'],8:['107','109','111'],20:['1007','1009','1011'],30:['1013','1015','1017']}

train_size=['100k','200k','500k']

for key in param_dict.keys():
    print(key)
    ndim=key
    sample_tag='true_ndims'+str(ndim)
    #true_data=generate_true_dataset(N_BKG=N_BKG,ndim=ndim)
    #SaveData(true_data,path_to_results,sample_tag)
    print(sample_tag)
    true_sample=load_sample(path_to_results,sample_tag)
    print(np.shape(true_sample))
    j=0
    for Model in param_dict.get(key):
        print(Model)
        sample_tag='nf_model'+str(Model)+'_trainsize'+str(train_size[j])+'_ndims'+str(ndim)
        print(sample_tag)
        #nf_sample=load_sample(path_to_results,sample_tag)
        #print(np.shape(nf_sample))
        nf_data=GenerateNFdata(N_SIG=N_SIG,iter_size=N_SIG/10,Model=Model)
        SaveData(nf_data,path_to_results,sample_tag)
        j=j+1



"""
time_start = time.time()
true_data=generate_true_dataset(N_BKG=N_BKG,ndim=ndim)
time_end = time.time()
sample_tag='true_ndims'+str(ndim)
SaveData(true_data,path_to_results,sample_tag)
print('time for true:'+str(time_end-time_start) )

time_start = time.time()
nf_data=GenerateNFdata(N_SIG=N_SIG,iter_size=N_SIG)
time_end = time.time()


print('time for nf:'+str(time_end-time_start) )



sample_tag='nf_model'+str(Model)+'_ndims'+str(ndim)
SaveData(nf_data,path_to_results,sample_tag)

loaded_data=load_sample(path_to_results,sample_tag)
print(np.shape(loaded_data))

"""