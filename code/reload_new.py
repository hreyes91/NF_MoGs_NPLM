import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfd = tfp.distributions
tfb= tfp.bijectors
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras import Model
from tensorflow.keras.callbacks import LambdaCallback
from scipy import stats
import sys
sys.path.append('../code')
import Distributions,Bijectors,MixtureDistributions
#import Trainer_2 as Trainer
import Metrics as Metrics
from statistics import mean,median
import pickle
import  matplotlib.pyplot as plt
import corner
from timeit import default_timer as timer
import os
import math


import GenerativeModelsMetrics as GMetrics
#import CorrelatedGaussians
#import Plotters
#import MixtureDistributions
#import compare_logprobs
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#import PlotsandHDPIforReload as PlotsandHDPI
import matplotlib.lines as mlines
#import Bijectors,Distributions,Metrics,MixtureDistributions,Plotters,Trainer,Utils


def MixtureGaussianDefined(path_npz,path,ncomp,ndims,seed=0):

    loc,scale,probs=LoadCMoGComp(path_npz)
    targ_dist = MixtureDistributions.MixMultiNormal1Def(path,loc,scale,probs,ncomp,ndims,seed=seed)
    return targ_dist


def LoadCMoGComp(path_npz):

    data = np.load(path_npz)
    # Access individual arrays
    loc = data['loc']
    scale = data['scale']
    probs = data['probs']

    return loc,scale,probs


def GenerateSinteticData(targ_dist,n_newsamples,path_npz,path,ncomp,ndims,seed=0):

    new_true_samples= targ_dist.sample(n_newsamples).numpy()

    return new_true_samples

def SaveSinteticData(new_true_samples,path,sample_tag):

    with open(path+'/true_sample_'+str(sample_tag)+'.npy', 'wb') as f:
        np.save(f, new_true_samples, allow_pickle=True)

    return

def GetTrueProb(targ_dist,samples):
    
    probs=targ_dist.prob(samples)

    return probs

def load_hyperparams(path_to_results):

    hyperparams_path=path_to_results+'/hyperparams.txt'
    hyperparams_frame=pd.read_csv(hyperparams_path)
    lastone=int(hyperparams_frame.shape[0]-1)

    ndims=int(hyperparams_frame['ndims'][lastone])
    nsamples=int(hyperparams_frame['nsamples'][lastone])
    bijector_name=str(hyperparams_frame['bijector'][lastone])
    nbijectors=int(hyperparams_frame['nbijectors'][lastone])
    batch_size=int(hyperparams_frame['batch_size'][lastone])
    spline_knots=int(hyperparams_frame['spline_knots'][lastone])
    range_min=int(hyperparams_frame['range_min'][lastone])
    activation=str(hyperparams_frame['activation'][lastone])
    regulariser=str(hyperparams_frame['regulariser'][lastone])
    eps_regulariser=float(hyperparams_frame['eps_regulariser'][lastone])
    hllabel=str(hyperparams_frame['hidden_layers'][lastone])
    
    
    hidden_layers=hllabel.split('-')
    for i in range(len(hidden_layers)):
        hidden_layers[i]=int(hidden_layers[i])

    return ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser




def ChooseBijector(bijector_name,nbijectors,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style='reverse'):


    if regulariser=='l1':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
    if regulariser=='l2':
        regulariser=tf.keras.regularizers.l2(eps_regulariser)
    else:
        regulariser=None
    
        

    #if bijector_name=='CsplineN':
    #    rem_dims=int(ndims/2)
    #    bijector=Bijectors.CsplineN(ndims,rem_dims,spline_knots,nbijectors,range_min,hidden_layers,activation)
    
    if bijector_name=='MsplineN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFNspline(ndims,spline_knots,nbijectors,range_min,hidden_layers,activation,kernel_initializer='glorot_uniform',kernel_regularizer=regulariser)
        


    if bijector_name=='MAFN':
        regulariser=tf.keras.regularizers.l1(eps_regulariser)
        bijector=Bijectors.MAFN(ndims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser,perm_style='reverse')
        
    if bijector_name=='RealNVPN':
        rem_dims=int(ndims/2)
        #regulariser=tf.keras.regularizers.l1(eps_regulariser)
        print(regulariser)
        print(eps_regulariser)

        bijector=Bijectors.RealNVPN(ndims,rem_dims,nbijectors,hidden_layers,activation,kernel_regularizer=regulariser,perm_style=perm_style)
    return bijector
    

def create_flow(bijector_name,nbijectors,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style='reverse'):
    
    bijector=ChooseBijector(bijector_name,nbijectors,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style=perm_style)
    base_dist=Distributions.gaussians(ndims)
    nf_dist=tfd.TransformedDistribution(base_dist,bijector)

    return nf_dist



def load_model(nf_dist,path_to_results,ndims,lr=.00001):

    x_ = Input(shape=(ndims,), dtype=tf.float32)
    
    model_temp = tf.keras.Model(inputs=x_, outputs=x_)
    x_tf = model_temp.layers[0].output
    
    log_prob_ = nf_dist.log_prob(x_)

    model = Model(x_, log_prob_)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr),
                loss=lambda _, log_prob: -log_prob)
    model.load_weights(path_to_results+'/model_checkpoint/weights')

    return nf_dist,model
    

        
@tf.function
def save_iter(nf_dist,sample_size,iter_size,n_iters):
    #first iter
    sample_all=nf_dist.sample(iter_size)
    for j in range(1,n_iters):
        if j%100==0:
            print(j/n_iters)
            #print(tf.shape(sample_all))
        sample=nf_dist.sample(iter_size)

        #sample=postprocess_data(sample,preprocess_params)
        sample_all=tf.concat([sample_all,sample],0)
        #if j%1==0:
        #    with open(path_to_results+'/nf_sample_5_'+str(j)+'.npy', 'wb') as f:
        #        np.save(f, sample, allow_pickle=True)
        #tf.keras.backend.clear_session()
    return sample_all



def save_sample(nf_dist,path_to_results,sample_size=100000,iter_size=10000,sample_tag='reload'):
    print('saving samples...')
    n_iters=int(sample_size/iter_size)
    

    sample_all=save_iter(nf_dist,sample_size,iter_size,n_iters)
    sample_all=sample_all.numpy()
    
    
    
    #print(np.shape(sample_all))
    with open(path_to_results+'/nf_sample_'+str(sample_tag)+'.npy', 'wb') as f:
        np.save(f, sample_all, allow_pickle=True)
    print('samples saved')
    return sample_all

def gen_sample(nf_dist,sample_size=100000,iter_size=10000):
    print('generating samples...')
    n_iters=int(sample_size/iter_size)
    

    sample_all=save_iter(nf_dist,sample_size,iter_size,n_iters)
    sample_all=sample_all.numpy()
    
    return sample_all
        
def load_sample(path_to_results):

    nf_sample=np.load(path_to_results+'/nf_sample.npy',allow_pickle=True)
    #nf_sample=np.load(path_to_results+'/sample_nf.pcl',allow_pickle=True)
    
    
    return nf_sample
        


def GetNFProb(nf_dist,samples):

    nf_probs=nf_dist.prob(samples)

    return nf_probs


def retrain_model(model,X_data,n_epochs,batch_size,patience=50,min_delta_patience=.00001,n_disp=1):

    ns = X_data.shape[0]
    if batch_size is None:
        batch_size = ns


    #earlystopping
    early_stopping=tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=min_delta_patience, patience=patience, verbose=1,
    mode='auto', baseline=None, restore_best_weights=False
     )
    # Display the loss every n_disp epoch
    epoch_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs:
                        print('\n Epoch {}/{}'.format(epoch+1, n_epochs, logs),
                              '\n\t ' + (': {:.4f}, '.join(logs.keys()) + ': {:.4f}').format(*logs.values()))
                                       if epoch % n_disp == 0 else False
    )


    checkpoint=tf.keras.callbacks.ModelCheckpoint(
    path_to_results+'/model_checkpoint/weights',
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode="auto",
    save_freq="epoch",
    options=None,

                )
                
                
    StopOnNAN=tf.keras.callbacks.TerminateOnNaN()



    history = model.fit(x=X_data,
                        y=np.zeros((ns, 0), dtype=np.float32),
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_split=0.3,
                        shuffle=True,
                        verbose=2,
                        callbacks=[epoch_callback,early_stopping,checkpoint,StopOnNAN])
    return history,nf_dist

def results_current(path_to_results,results_dict):

    currrent_results_file=open(path_to_results+'results_reload_test.txt','w')
    header=','.join(list(results_dict.keys()))



    currrent_results_file.write(header)
    currrent_results_file.write('\n')
    
    string_list=[]
    for key in results_dict.keys():
        string_list.append(str(results_dict.get(key)[-1]))
    
    string=','.join(string_list)
    currrent_results_file.write(string)
    currrent_results_file.write('\n')
    
    currrent_results_file.close()


    return



    
def SaveNewMetrics(metrics_dict,metric_name,results_path):

    name=results_path+'/'+metric_name+'_dict_sf'
    np.save(name,metrics_dict)


    return


def SaveResultsCurrentNewMetrics(results_dict_newmetrics,path_to_results):

    
    Frame=pd.DataFrame(results_dict_newmetrics)
    current_row= Frame.tail(1)
    current_row.to_csv(path_to_results+'/results_new_metrics.txt',index=False)

    return


def SaveMeans(test_means):

    print(pd.DataFrame(test_means))

    return


def GetReusltsForPOI(results_list,labels,path_to_results,name):

    results_list=np.transpose(results_list)

    
    POI_results_mean=np.mean(results_list,axis=1)

    POI_results_mean_frame=pd.DataFrame(POI_results_mean).transpose()
    POI_results_mean_frame.columns =labels
    POI_results_mean_frame.to_csv(path_to_results+'/'+name+'_POI_results.txt',index=False)
    
    return POI_results_mean.tolist()

def ResultsToDict_NewMetrics(KS_pv_mean,KS_pv_std,KS_st_mean,KS_st_std,SWDmean,SWDstd):
    """
    Function that writes results to the a dictionary.
    """
    results_dict_newmetrics.get('KS_pv_mean').append(KS_pv_mean)
    results_dict_newmetrics.get('KS_pv_std').append(KS_pv_std)
    results_dict_newmetrics.get('KS_st_mean').append(KS_st_mean)
    results_dict_newmetrics.get('KS_st_std').append(KS_st_std)

    results_dict_newmetrics.get('SWD_mean').append(SWDmean)
    results_dict_newmetrics.get('SWD_std').append(SWDstd)

    return results_dict_newmetrics
    
def cornerplotter(target_samples,nf_samples,path_to_plots,ndims,tag='reload', save_plot=False):


    n_bins = 50
    red_bins=50
    
    density=(np.max(target_samples,axis=0)-np.min(target_samples,axis=0))/red_bins
    
    blue_bins=(np.max(nf_samples,axis=0)-np.min(nf_samples,axis=0))/density
    blue_bins=blue_bins.astype(int).tolist()


    blue_line = mlines.Line2D([], [], color='red', label='target')
    red_line = mlines.Line2D([], [], color='blue', label='NF')
    figure=corner.corner(target_samples,color='red',bins=red_bins)
    corner.corner(nf_samples,color='blue',bins=blue_bins,fig=figure)
    plt.legend(handles=[blue_line,red_line], bbox_to_anchor=(-ndims+1.8, ndims+.3, 1., 0.) ,fontsize='xx-large')
    if save_plot:
        plt.savefig(path_to_plots+'/corner_plot_'+tag+'.pdf')
        plt.close()
    else:
        plt.show()
    return

def MeansAndStdMetrics(metric_result):

    metric_mean=np.mean(metric_result)
    metric_std=np.std(metric_result)

    return metric_mean,metric_std
    
def SaveNewMetrics(metrics_dict,metric_name,results_path, tag='reload_dict_sf'):

    name=results_path+'/'+metric_name+tag
    print(name)
    np.save(name,metrics_dict)


    return

##### sample
def sample_nf(n,path_to_model):
    ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser=load_hyperparams(path_to_model)
    nf_arq=create_flow(bijector_name,nbijectors,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style='reverse')
    nf_dist,model=load_model(nf_arq,path_to_model,ndims,lr=.00001)
    
    return gen_sample(nf_dist,sample_size=n,iter_size=1000)

def sample_true(n,ndims,path_to_results='results/test_2_RealNVPN/',seed=0,ncomp=3):
    path_npz=path_to_results+'CMoG_params_ndims'+str(ndims)+'_ncomp_'+str(ncomp)+'.npz'
    targ_dist = MixtureGaussianDefined(path_npz,path_to_results,ncomp,ndims,seed=seed)
    
    return GenerateSinteticData(targ_dist,n,path_npz,path_to_results,ncomp,ndims,seed=seed)


##### sample and prob
def sample_nf_prob(n,path_to_model):
    ndims,nsamples,bijector_name,nbijectors,batch_size,spline_knots,range_min,activation,hidden_layers,hllabel,regulariser,eps_regulariser=load_hyperparams(path_to_model)
    nf_arq=create_flow(bijector_name,nbijectors,ndims,spline_knots,range_min,hidden_layers,activation,regulariser,eps_regulariser,perm_style='reverse')
    nf_dist,_=load_model(nf_arq,path_to_model,ndims,lr=.00001)
    
    sample = gen_sample(nf_dist,sample_size=n,iter_size=1000)
    
    sample_tag = 'reload_nsamples'+str(n)
    nf_samples = save_sample(nf_dist,path_to_model,sample_size=n,iter_size=1000,sample_tag=sample_tag)
    
    probs = GetNFProb(nf_dist,nf_samples)
    
    return sample, probs


def sample_true_prob(n,ndims,path_to_results='results/test_2_RealNVPN/',seed=0,ncomp=3):
    path_npz=path_to_results+'CMoG_params_ndims'+str(ndims)+'_ncomp_'+str(ncomp)+'.npz'
    targ_dist = MixtureGaussianDefined(path_npz,path_to_results,ncomp,ndims,seed=seed)
    
    sample = GenerateSinteticData(targ_dist,n,path_npz,path_to_results,ncomp,ndims,seed=seed)
    probs = GetTrueProb(targ_dist,sample) 
    
    return sample, probs


'''CORRELATION'''
def MixtureGaussian_corr(path,ncomp,ndims,seed=0):
    targ_dist = MixtureDistributions.RealCMog(path,ncomp,ndims,seed=seed)
    return targ_dist

def LoadCMoGComp_corr(path_npz):

    data = np.load(path_npz)
    # Access individual arrays
    means = data['means']
    matrix_elements_list = data['matrix_elements_list']
    weights = data['weights']
    return means,matrix_elements_list,weights

def MixtureGaussianDefined_corr(path_npz,path,ncomp,ndims,seed=0):

    means,matrix_elements_list,weights=LoadCMoGComp_corr(path_npz)
    print('hello loc')
    targ_dist = MixtureDistributions.MixMultiNormal1Def(path,means,matrix_elements_list,weights,ncomp,ndims,seed=seed)
    return targ_dist


def sample_true_corr(n,ndims,path_to_results='results/test_2_RealNVPN/',seed=0,ncomp=3):
    path_npz=path_to_results+'/'+'RealCMoG_params_ndims'+str(ndims)+'_ncomp_'+str(ncomp)+'.npz'
    targ_dist=MixtureGaussian_corr(path_to_results,ncomp,ndims,seed=seed)
    # targ_dist = MixtureGaussianDefined_corr(path_npz,path_to_results,ncomp,ndims,seed=seed)
    X_data_test=targ_dist.sample(n).numpy()
    return X_data_test

