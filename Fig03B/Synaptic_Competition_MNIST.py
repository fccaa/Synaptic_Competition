#!/usr/bin/env python
# coding: utf-8

#
# Loading libraries
#

import numpy as np

from sklearn.datasets import fetch_openml
import joblib
import sys
import daz

daz.set_daz()
daz.set_ftz()

#
# Fetching the dataset from server and scale 
# it for our purpose
#

# Fetch the datasets

mnist = fetch_openml("mnist_784")

# Normalize the dataset
mnist_data = mnist.data.values / np.max(mnist.data.values)

# Get Targets
mnist_target = mnist.target.astype(int)

print("Data Ready", file = sys.stderr , flush=True)

# Convert labels to binaries
mnist_label = np.zeros((mnist_data.shape[0], 10))
for idx in range(mnist_data.shape[0]):
    mnist_label[idx,int(mnist.target[idx])] = 1

# Clean memory space of mnist
mnist = None

# The activation function for y

def y_of_x(x:np.ndarray, W_E_xy:np.ndarray = None, 
           W_I_yy:np.ndarray = None, noise = None
            ):
    
    if W_E_xy is None:
        return 0.5*(x + np.abs(x))

    if len(x.shape) == 1:
        y = np.zeros((W_E_xy.shape[0]))
    else:
        y = np.zeros((W_E_xy.shape[0], x.shape[1]))

    for t in range(20):
        input4y = np.matmul(W_E_xy, x) / np.sqrt(W_E_xy.shape[1])
        if W_I_yy is not None:
            input4y -= np.matmul(W_I_yy, (y)) / np.sqrt(W_I_yy.shape[1])
        if noise is not None:
            input4y += noise
        y += (-y + np.maximum(input4y, 0)) / np.sqrt(5.0)

    return y

def update_w(x, w, cut = False):
    
    w_local = np.copy(w)
    
    x_last = np.zeros(w.shape[1])
    y_last = np.zeros(w.shape[0])
    
    for idx in range(x.shape[0]):
        
        xt = x[idx]
        
#        noise = np.heaviside(np.random.rand(w_local.shape[0]) - 0.95, 0)
        noise = np.random.normal(size=w_local.shape[0])
    
        y = y_of_x(xt, w_local, noise=noise)

        delta_w = np.outer(y, xt)
        
        for i in range(w_local.shape[0]):
            if np.sum(w_local[i]) >= 1.0:
                delta_w[i] = np.zeros_like(delta_w[i])
                
        for i in range(delta_w.shape[1]):
            delta_w[:,i] -= np.mean(delta_w[:,i])

        w_local += 0.1 * delta_w
                           
        for i in range(w_local.shape[1]):
            if np.max(w_local[:,i]) > 1:
                w_local[:,i] /= np.max(w_local[:,i])

        for i in range(w_local.shape[0]):
            if np.max(w_local[i]) < 0:
                w_local[i] = np.zeros_like(w_local[i])
              
    return w_local

def cal_fit_rate(W_E_xy:np.ndarray, 
                 data_sample:np.ndarray, target_sample:np.ndarray,
                 data_test:np.ndarray, target_test:np.ndarray):

    y = y_of_x(data_sample.transpose(), W_E_xy)

    y_tmp = np.append(y, np.ones((1,y.shape[1])), axis=0)

    y = y_tmp

    W_yz = np.matmul(y, y.transpose())

    W_yz = np.matmul(y.transpose(), np.linalg.pinv(W_yz))

    W_yz = np.matmul(target_sample.transpose(), W_yz)

    y = y_of_x(data_test.transpose(), W_E_xy)

    y_tmp = np.append(y, np.ones((1,y.shape[1])), axis=0)

    y = y_tmp

    z = np.matmul(W_yz, y)

    ans = 1.0 - np.sum(np.heaviside(
                    np.absolute(z.argmax(axis=0) - target_test.argmax(axis=1)) - 0.5
                    , 0)) / float(data_test.shape[0])

    return ans

def local_loop(mnist_data, mnist_label, N_train_sample, N_test, N_y):

    random_list = np.arange(mnist_data.shape[0])

    np.random.shuffle(random_list)

    mnist_target_training = mnist_label[random_list[:N_train_sample]]
    mnist_data_training = mnist_data[random_list[:N_train_sample]]

    mnist_target_testing = mnist_label[random_list[(mnist_data.shape[0]-N_test):]]
    mnist_data_testing = mnist_data[random_list[(mnist_data.shape[0]-N_test):]]

    W_E_xy = np.zeros((N_y, mnist_data.shape[1]))

    W_E_xy = update_w(mnist_data_training, W_E_xy)

    return cal_fit_rate(W_E_xy, mnist_data_training, mnist_target_training,
                 mnist_data_testing, mnist_target_testing)

# Number of middle layer

N_y = int(100)

# Number of training samples 

N_train_sample = int(500)

for N_y in np.arange(100, 1001, 100):

    for N_train_sample in [100,500,1000,5000,10000,50000]:

        #
        # Cutting the dataset to different matices
        #
        
        # Copy the dataset to training and testing
        
        N_test = int(mnist_data.shape[0] * 0.1)
        
#         mnist_target_training = mnist_label[random_list[:N_train_sample]]
#         mnist_data_training = mnist_data[random_list[:N_train_sample]]
        
#         mnist_target_testing = mnist_label[random_list[(mnist_data.shape[0]-N_test):]]
#         mnist_data_testing = mnist_data[random_list[(mnist_data.shape[0]-N_test):]]
        
        # Running the training
        
        for rate in [0.1]:
            for w_max in [1.0]:
                sample_ans = []
                
#                 for idx in range(10):
        
#                     random_list = np.arange(mnist_data.shape[0])
        
#                     np.random.shuffle(random_list)
                    
#                     mnist_target_training = mnist_label[random_list[:N_train_sample]]
#                     mnist_data_training = mnist_data[random_list[:N_train_sample]]
                    
#                     mnist_target_testing = mnist_label[random_list[(mnist_data.shape[0]-N_test):]]
#                     mnist_data_testing = mnist_data[random_list[(mnist_data.shape[0]-N_test):]]
                    
#                     W_E_xy = np.zeros((N_y, mnist_data.shape[1]))
                    
#                     W_E_xy = update_w(mnist_data_training, W_E_xy)
        
#                     sample_ans.append(cal_fit_rate(W_E_xy, mnist_data_training, mnist_target_training,
#                                  mnist_data_testing, mnist_target_testing))

#                     sample_ans.append(local_loop(mnist_data, mnist_label, N_train_sample, N_test, N_y))

            
                sample_ans = joblib.Parallel(n_jobs=-1)(joblib.delayed(local_loop)(mnist_data, mnist_label, N_train_sample, N_test, N_y)
                                                        for idx in range(10))
        
                print(N_y, N_train_sample, np.mean(sample_ans), np.std(sample_ans) , flush=True)

    print(flush=True)






exit(0)

#
# Plot the weightes
#

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

N_plot = 100

plt.figure(figsize=(15, 50))

for idx in np.arange(N_plot):

    ax = plt.subplot(20,5,idx+1)

    im = ax.imshow(W_E_xy[idx].reshape((28,28)))
    
    divider = make_axes_locatable(ax)
    
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax=cax)





