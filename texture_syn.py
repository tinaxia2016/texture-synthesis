# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:33:19 2018

@author: xiaji
"""

from skimage.io import imread, imsave
from os.path import normpath as fn
import numpy as np
import matplotlib.pyplot as plt

##initialize output image parameters
img = np.float32(imread(fn('input/paper.jpg')))/255.
img = img[:100,:100,:]
H, W, C = img.shape

block_size = 20
n_patch = 10
overlap_width = 5
#
#output_size = int((block_size-overlap_width)*n_patch + overlap_width)  #10x10 blocks
#
#output = np.zeros((output_size,output_size,3))
#y = np.random.randint(0,H-block_size)
#x = np.random.randint(0,W-block_size)
#output[:block_size, :block_size,:] = img[y:y+block_size, x:x+block_size,:]


##
def left_error(img,output,ind_pc):
    left_error = np.zeros((H-block_size,W-block_size))
    output_overlap = output[ind_pc[0]*(block_size-overlap_width): ind_pc[0]*(block_size-overlap_width)+block_size,\
                            ind_pc[1]*(block_size-overlap_width): ind_pc[1]*(block_size-overlap_width)+overlap_width,:]
    for i in range(H-block_size):
        for j in range(W-block_size):
            img_overlap = img[i: i+block_size, j: j+overlap_width,:]
            left_error[i,j] = np.sum((img_overlap - output_overlap)**2)

    return left_error

def up_error(img,output,ind_pc):
    up_error = np.zeros((H-block_size,W-block_size))
    output_overlap = output[ind_pc[0]*(block_size-overlap_width): ind_pc[0]*(block_size-overlap_width)+overlap_width,\
                            ind_pc[1]*(block_size-overlap_width): ind_pc[1]*(block_size-overlap_width)+block_size,:]
    for i in range(H-block_size):
        for j in range(W-block_size):
            img_overlap = img[i: i+overlap_width, j: j+block_size,:]
            up_error[i,j] = np.sum((img_overlap - output_overlap)**2)

    return up_error
  
#generate a 500x500 image with 10x10 patch
def best_patch(ind_pc):
    #ind_pc is the index of patch, e.g. [0,1] means first row, second column patch
    if (ind_pc[0]==0) & (ind_pc[1]>0):
        error = left_error(img,output,ind_pc)
    elif (ind_pc[0]>0) & (ind_pc[1]==0):
        error = up_error(img, output, ind_pc)
    else:
        error = up_error(img,output,ind_pc)+left_error(img,output,ind_pc)
    
    error[error==0] = 100
    
    min_error = np.min(error)
    px_list = np.where(error<=min_error*1.1)
    
    return px_list


def vertical_cost(patch, output, ind_pc):
    
    vertical_cost = np.zeros((block_size,overlap_width))
    output_overlap = output[ind_pc[0]*(block_size-overlap_width): ind_pc[0]*(block_size-overlap_width)+block_size,\
                            ind_pc[1]*(block_size-overlap_width): ind_pc[1]*(block_size-overlap_width)+overlap_width,:]
    
    patch_overlap = patch[0: block_size, 0: overlap_width,:]
    
    vertical_cost = np.sum((output_overlap-patch_overlap)**2,axis=2)
    
    return vertical_cost

def horizontal_cost(patch, output, ind_pc):
    
    horizontal_cost = np.zeros((overlap_width,block_size))
    output_overlap = output[ind_pc[0]*(block_size-overlap_width): ind_pc[0]*(block_size-overlap_width)+overlap_width,\
                            ind_pc[1]*(block_size-overlap_width): ind_pc[1]*(block_size-overlap_width)+block_size,:]
    
    patch_overlap = patch[0: overlap_width, 0: block_size,:]
    
    horizontal_cost = np.sum((output_overlap-patch_overlap)**2,axis=2)
    
    return horizontal_cost


def vertical_cut(vertical_cost):
    
    d = np.zeros((block_size, overlap_width),dtype=int)
    d[0,:] = np.arange(overlap_width,dtype=int)
    
    c_err = np.zeros((block_size, overlap_width))
    c_err[0,:] = vertical_cost[0,:]
    
    vertical_cut = np.zeros((block_size,),dtype=int)
    
    for i in range(1,block_size):
        for j in range(overlap_width):
            if j==0:
                d[i,j] = j + np.argmin([c_err[i-1,j], c_err[i-1,j+1]])
                
            
            elif j==overlap_width-1:
                d[i,j] = j - 1 + np.argmin([c_err[i-1,j-1], c_err[i-1,j]])
               
            else:
                d[i,j] = j -1 + np.argmin([c_err[i-1,j-1], c_err[i-1,j], c_err[i-1,j+1]])
            
            c_err[i,j] = c_err[i-1,d[i,j]] + vertical_cost[i,j]
    
    ind = np.argmin(c_err[block_size-1,:])
    for i in range(block_size-1,-1,-1):
        vertical_cut[i] = ind
        ind = d[i,ind]
    
    return vertical_cut


def horizontal_cut(horizontal_cost):
    
    d = np.zeros((overlap_width, block_size),dtype=int)
    d[:,0] = np.arange(overlap_width,dtype=int)
    
    c_err = np.zeros((overlap_width, block_size))
    c_err[:,0] = horizontal_cost[:,0]
    
    horizontal_cut = np.zeros((block_size,),dtype=int)
    
    for i in range(1,block_size):
        for j in range(overlap_width):
            if j==0:
                d[j,i] = j + np.argmin([c_err[j,i-1], c_err[j+1,i-1]])
                
            
            elif j==overlap_width-1:
                d[j,i] = j - 1 + np.argmin([c_err[j-1,i-1], c_err[j,i-1]])
               
            else:
                d[j,i] = j -1 + np.argmin([c_err[j-1,i-1], c_err[j,i-1], c_err[j+1,i-1]])
            
            c_err[j,i] = c_err[d[j,i],i-1] + horizontal_cost[j,i]
    
    ind = np.argmin(c_err[:,block_size-1])
    for i in range(block_size-1,-1,-1):
        horizontal_cut[i] = ind
        ind = d[ind,i]
    
    return horizontal_cut


def quilt_vertical(vertical_cut,patch, ind_pc):
    for i in range(block_size):
        patch[i,:vertical_cut[i],:] = output[ind_pc[0]*(block_size-overlap_width)+i, \
                                           ind_pc[1]*(block_size-overlap_width):vertical_cut[i] + ind_pc[1]*(block_size-overlap_width),:]
    return patch

def quilt_horizontal(horizontal_cut,patch, ind_pc):
    for i in range(block_size):
        patch[:horizontal_cut[i],i,:] = output[ind_pc[0]*(block_size-overlap_width):horizontal_cut[i] + ind_pc[0]*(block_size-overlap_width), \
                                              ind_pc[1]*(block_size-overlap_width)+i,:]
    return patch
 
    

output_size = int((block_size-overlap_width)*n_patch + overlap_width)  #10x10 blocks

output = np.zeros((output_size,output_size,3))
y = np.random.randint(0,H-block_size)
x = np.random.randint(0,W-block_size)
output[:block_size, :block_size,:] = img[y:y+block_size, x:x+block_size,:]

for i in range(n_patch):
    for j in range(n_patch):
        ind_pc = [i,j]
        print(i,j)

        if (i!=0) | (j!=0):
            px_list = best_patch(ind_pc)
            rand_pick = np.random.randint(len(px_list[0]))
            x1 = px_list[0][rand_pick]
            y1 = px_list[1][rand_pick]
            
            patch = img[x1:x1+block_size,y1:y1+block_size,:].copy()
            
            if i==0:
            #only need vertical quilt
                vcost = vertical_cost(patch,output, ind_pc) 
                vcut = vertical_cut(vcost)
                patch_new = quilt_vertical(vcut,patch,ind_pc)

            elif j==0:
            #only need horizontal quilt
                hcost = horizontal_cost(patch,output, ind_pc) 
                hcut = horizontal_cut(hcost)
                patch_new = quilt_horizontal(hcut,patch,ind_pc)
                
            else:
                vcost = vertical_cost(patch,output, ind_pc) 
                vcut = vertical_cut(vcost)
                patch_new = quilt_vertical(vcut,patch,ind_pc)

                hcost = horizontal_cost(patch_new,output, ind_pc) 
                hcut = horizontal_cut(hcost)
                patch_new = quilt_horizontal(hcut,patch_new,ind_pc)

            output[i*(block_size-overlap_width):i*(block_size-overlap_width)+block_size, \
                   j*(block_size-overlap_width):j*(block_size-overlap_width)+block_size] = patch_new
                

f, (a0, a1) = plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 3]})

a0.imshow(img)

a1.imshow(output)
plt.show()



