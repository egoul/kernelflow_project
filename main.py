
# coding: utf-8

# In[1]:

import tensorflow as tf
import sys
from keras.losses import binary_crossentropy
from keras.layers import Dense, InputLayer
from keras.models import Sequential
import h5py

sys.path.append("../kernelflow/")

import numpy as np
from numpy.random import multivariate_normal as mvn
import matplotlib.pyplot as plt

from kernelflow.kernel_density import KernelDensity


#import dataset
train = h5py.File('m100_data/m100_train.h5', 'r')
test = h5py.File('m100_data/m100_test.h5', 'r')
trainX = np.asarray(train['features'][()])
trainY = np.asarray(train['targets'][()])
trainW = np.asarray(train['weights'][()])
testX = np.asarray(test['features'][()])
testY = np.asarray(test['targets'][()])
testW = np.asarray(train['weights'][()])


#generate s_dat, b_dat, s_val, b_val
s_dat = np.asarray([trainX[i] for i in range(len(trainX)) if trainY[i] == 1])

s_dat_w = np.asarray([trainW[i] for i in range(len(trainW)) if trainY[i] == 1])
b_dat = np.asarray([trainX[i] for i in range(len(trainX)) if trainY[i] == 0])
b_dat_w = np.asarray([trainW[i] for i in range(len(trainW)) if trainY[i] == 0])

s_val = np.asarray([testX[i] for i in range(len(testX)) if testY[i] == 1])
#s_val_w = np.asarray([testW[i] for i in range(len(testW)) if testY[i] == 1])
b_val = np.asarray([testX[i] for i in range(len(testX)) if testY[i] == 0])
#b_val_w = np.asarray([testW[i] for i in range(len(testW)) if testY[i] == 0])





# In[4]:

# keras+tf dnn model

n_features = 11
thresh = 0.5

m_input = tf.placeholder(tf.float32, shape=(None, n_features))
s_weights = tf.placeholder(tf.float32, shape=(None, 1))
#s_weights = np.concat((s_dat_w,b_dat_w))
sample_mass = tf.placeholder(tf.float32,shape=(1))

model = Sequential()
model.add(InputLayer(input_tensor=m_input,input_shape = (None, n_features)))
model.add(Dense(32, activation='relu')) #two relu layers w/ 32 neurons
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

m_weights = model.get_weights()
m_output = model.output

labels = tf.placeholder(tf.float32, shape=(None, 1))

mu_asimov = tf.placeholder(tf.float32, shape=(1,))
mu_float = tf.placeholder(tf.float32, shape=(1,))


# In[5]:

# batch generators
#can control proportion of signal and background
#and the amount of signal per batch
def random_split_gen(batch_size, sig_frac_batch=0.5,
                     return_weights = True,
                     s_sum_weight=100.,b_sum_weight=1000.):

    s_batch = int(batch_size*sig_frac_batch)
    b_batch = batch_size-s_batch
    n_s_batch = s_dat.shape[0]//s_batch
    n_b_batch = b_dat.shape[0]//b_batch
    n_batch = min(n_s_batch,n_b_batch)
    yield n_batch # yield first nb_epoch first
    while True:
        s_ids = np.random.permutation(s_dat.shape[0]) #permutation of signal indices
        b_ids= np.random.permutation(b_dat.shape[0]) #permutation of bg indices
        for b_n in range(n_batch): #for each batch
            b_input = np.zeros((batch_size,n_features)) #create empty placeholder input array
            b_output = np.zeros((batch_size,1)) #create empty placeholder output array
            b_weight = np.zeros((batch_size,1)) #create empty placeholder weight array
            s_mask = s_ids[b_n*s_batch:(b_n+1)*s_batch] #get the current batch of signal indices
            b_input[:s_batch] = s_dat[s_mask] #put current signal batch in input array
            b_output[:s_batch] = 1. #put output for signal as 1
            b_weight[:s_batch] = s_sum_weight/float(s_batch) #build weights
            b_mask = b_ids[b_n*b_batch:(b_n+1)*b_batch] #get current batch of bg indices
            b_input[s_batch:batch_size] = b_dat[b_mask] #put current bg batch in array, immediately after signal
            b_output[s_batch:batch_size] = 0. #put output for bg as 0
            b_weight[s_batch:batch_size] = b_sum_weight/float(b_batch) #get bg weights
            if return_weights:
                yield (b_input, b_output, b_weight)
            else:
                yield (b_input, b_output)



# In[6]:

init = tf.global_variables_initializer()


# In[12]:

bw = 0.001
cut = 0.5
is_sig = tf.cast(labels, dtype=tf.bool)
is_bkg = tf.logical_not(is_sig)
sig_output = tf.boolean_mask(m_output, is_sig) #grab the signal data
bkg_output = tf.boolean_mask(m_output, is_bkg) #grab the bg data
sig_weights = tf.boolean_mask(s_weights, is_sig) #grab the signal weights
bkg_weights = tf.boolean_mask(s_weights, is_bkg) #grab the bg weights
s_kde = KernelDensity(tf.reshape(sig_output,[-1]), #create a signal kernel density centered at the outputs and weighted
                      [bw], sig_weights )
b_kde = KernelDensity(tf.reshape(bkg_output,[-1]), #create a bg kernel density centered at the outputs and weighted
                      [bw], bkg_weights)
s_log_count = s_kde.log_cdf(cut) #get the log count for signal?
b_log_count = b_kde.log_cdf(cut) #get the log count for bg


# custom poison prob so it is defined for non-integers
def log_prob_poisson(x,rate):
    return x * tf.log(rate) - tf.lgamma(x + 1.)-rate

# differenciable poisson asimov hessian based variance loss
def asimov_likelihood(mu_asimov, mu_float, cut=0.5, bw=0.01):
    is_sig = tf.cast(labels, dtype=tf.bool) #get signal and background output values and weights
    is_bkg = tf.logical_not(is_sig)
    sig_output = tf.boolean_mask(m_output, is_sig)
    bkg_output = tf.boolean_mask(m_output, is_bkg)
    sig_weights = tf.boolean_mask(s_weights, is_sig)
    bkg_weights = tf.boolean_mask(s_weights, is_bkg)
    s_kde = KernelDensity(tf.reshape(sig_output,[-1]), #get kernel densities objects for signal and background
                          [bw], sig_weights )
    b_kde = KernelDensity(tf.reshape(bkg_output,[-1]),
                          [bw], bkg_weights)
    s_log_count = s_kde.log_cdf(cut)
    b_log_count = b_kde.log_cdf(cut)
    #calculate the likelihood
    return log_prob_poisson(mu_asimov*tf.reduce_sum(sig_weights)*tf.exp(s_log_count)+ #x = mu_asimov * s + b
                            tf.reduce_sum(bkg_weights)*tf.exp(b_log_count),
                            mu_float*tf.reduce_sum(sig_weights)*tf.exp(s_log_count)+ #rate = mu_float * s + b
                            tf.reduce_sum(bkg_weights)*tf.exp(b_log_count))

#approximation of the significance with the significance,
#only valid under specific conditions
def simple_sig(cut=0.5, bw=0.01):
    is_sig = tf.cast(labels, dtype=tf.bool) #grab signal and background outputs and weights, build corresponding kdes
    is_bkg = tf.logical_not(is_sig)
    sig_output = tf.boolean_mask(m_output, is_sig)
    bkg_output = tf.boolean_mask(m_output, is_bkg)
    sig_weights = tf.boolean_mask(s_weights, is_sig)
    bkg_weights = tf.boolean_mask(s_weights, is_bkg)
    s_kde = KernelDensity(tf.reshape(sig_output,[-1]),
                          [bw], sig_weights )
    b_kde = KernelDensity(tf.reshape(bkg_output,[-1]),
                          [bw], bkg_weights)
    s_log_count = s_kde.log_cdf(cut)
    b_log_count = b_kde.log_cdf(cut)
    return tf.reduce_sum(sig_weights)*tf.exp(s_log_count)/tf.sqrt(tf.reduce_sum(bkg_weights)*tf.exp(b_log_count))


simple_sig_no_cut = simple_sig(0.0,1e-6) #simple sig results from kde with different cuts
simple_sig_half = simple_sig(0.5,1e-6)
asimov_mu_ll = asimov_likelihood(mu_asimov,mu_float,cut=0.5, bw=0.1) #bw = 0.1 estimates from kde
asimov_zero_ll = asimov_likelihood(mu_asimov,0,cut=0.5, bw=0.1)
sig_ll_ratio = tf.sqrt(-2.*(asimov_zero_ll-asimov_mu_ll))[0]
asimov_mu_ll_exact = asimov_likelihood(mu_asimov,mu_float,cut=0.5, bw=1.e-6)  #bw = 1e-6 estimates from kde
asimov_zero_ll_exact = asimov_likelihood(mu_asimov,0,cut=0.5, bw=1.e-6)
sig_ll_ratio_exact = tf.sqrt(-2.*(asimov_zero_ll_exact-asimov_mu_ll_exact))[0]

#mu_ll is the fraction of signal, want to fit mu
hess = - tf.hessians(asimov_mu_ll, mu_float)[0] #-hessian of asimov_mu_ll w.r.t mu_float
print('hess ',hess)
inv_hess = tf.matrix_inverse(hess)[0] #inverse of -hessian

#gaussian approximation of the significance
sig_hess = tf.sqrt(mu_asimov**2/inv_hess)[0] # sqrt(mu^2/H^{-1})

#creates a histogram of the data by integrating the kde from the datapoints in each bin
def KDEHist(binpoints, kde,start):
    #assume that binpoints includes start, end, and intermediates
    binpoints = tf.reshape(binpoints,[-1,1])
    lefts = kde.cdf(binpoints[:-1])
    rights = kde.cdf(binpoints[1:])
    counts = tf.subtract(rights,lefts)
    return (binpoints, counts)

testdata = tf.placeholder(tf.float32,shape=(None,n_features))
preds = tf.placeholder(tf.float32,shape=(None))
KDE1 = KernelDensity(preds, [bw],weight=None)
KDE1w = KDE1.cdf(preds)
masses = testdata[:,7]
sig_mask = tf.greater_equal(preds,thresh)
bg_mask = tf.less(preds,thresh)
sig = tf.boolean_mask(tf.reshape(masses,[-1,1]),tf.reshape(sig_mask,[-1,1]))
sigw = tf.reshape(tf.boolean_mask(KDE1w,tf.tile(sig_mask,[1,1])),[-1])
bg = tf.boolean_mask(tf.reshape(masses,[-1,1]),tf.reshape(bg_mask,[-1,1]))
bgw = tf.boolean_mask(KDE1w,tf.tile(bg_mask,[1,1]))
sigKDE = KernelDensity(sig,[bw],sigw)
bgKDE = KernelDensity(bg,[bw],bgw)

bin_width = .1
start = 0.
end = tf.add(tf.reduce_max(masses),bin_width)
binpoints = tf.reshape(tf.range(start, end, bin_width),[-1])
sig_hist = KDEHist(binpoints,sigKDE,start)
bg_hist = KDEHist(binpoints, bgKDE,start)

# In[8]:

def better_sig(s,b):
    return np.sqrt(2.*((s+b)*np.log(1+s/b)-s))


# In[14]:

asimov_sig_loss = - sig_ll_ratio
#train to maximize sig ll ratio
train_asimov_sig = tf.train.GradientDescentOptimizer(0.1).minimize(asimov_sig_loss, var_list=model.weights)

n_epoch = 2
rs_gen = random_split_gen(8192)
n_batch = rs_gen.__next__()

rs_val_gen = random_split_gen(8192) #batch size of 8192
n_batch_val = rs_val_gen.__next__()
v_input, v_output, v_weight = rs_val_gen.__next__()
v_is_sig = (v_output == 1) #get masks
v_is_bkg = (v_output == 0)

wps = np.linspace(0.1,0.9,20)

# initialize variables

simple_sig_arr = np.zeros((n_epoch,n_batch))
bce_arr = np.zeros((n_epoch,n_batch))
sig_ll_ratio_arr = np.zeros((n_epoch,n_batch))
sig_ll_ratio_exact_arr = np.zeros((n_epoch,n_batch))
sig_hess_arr = np.zeros((n_epoch,n_batch))
sig_ll_ratio_val_arr = np.zeros((n_epoch, wps.shape[0]))
sb_test = []
sb_pred = []
logs_path = ""

with tf.Session() as sess:
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    sess.run([init])
    for e_n in range(n_epoch):
        for b_n in range(n_batch):
            b_input, b_output, b_weight = rs_gen.__next__()
            var_dict = {m_input : b_input,
                labels: b_output,
                s_weights : b_weight,
                mu_asimov : [1.],
                mu_float : [1.]}
            sess.run([train_asimov_sig], var_dict)
            simple_sig_arr[e_n,b_n] = simple_sig_half.eval(var_dict)
            sig_ll_ratio_arr[e_n,b_n] = sig_ll_ratio.eval(var_dict)
            sig_ll_ratio_exact_arr[e_n,b_n] = sig_ll_ratio_exact.eval(var_dict)

        pred_val = m_output.eval({m_input: v_input})
        for val in pred_val:
            sb_pred.append(val)
        for dat in v_input:
            sb_test.append(dat)
        for w_n, wp in enumerate(wps):
            s = np.sum(v_weight[v_is_sig][pred_val[v_is_sig] > wp])
            b = np.sum(v_weight[v_is_bkg][pred_val[v_is_bkg] > wp])
            sig_ll_ratio_val_arr[e_n,w_n] = better_sig(s,b)
    #training
    sb_test = np.asarray(sb_test)
    sb_pred = np.asarray(sb_pred)
    in_dict = {testdata: sb_test, preds: sb_pred}
