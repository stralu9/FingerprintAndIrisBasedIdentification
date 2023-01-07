# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 22:50:30 2016

@author: utkarsh
"""
from .ridge_segment import ridge_segment
from .ridge_orient import ridge_orient
from .ridge_freq import ridge_freq
from .ridge_filter import ridge_filter
from .ridge_segment import largest_connected_component
from .segment import segment
import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from skimage.filters import sobel
from scipy import ndimage as ndi


def Sobel_mask(img):
    thresh = sobel(img)    
    img = cv2.dilate(thresh, None, iterations=5)
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN,None, iterations=5)
    edge = np.array(img)*255
    edge = np.where(edge > 50, 255, 0)
    edge = edge.astype(np.uint8)
    cnt = sorted(cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    mask = cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask

def Canny_mask(img):
    thresh = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    edge = cv2.dilate(cv2.Canny(img, 0, 255), None, iterations=5)
    img = cv2.morphologyEx(edge,cv2.MORPH_OPEN,None, iterations=5)

    
    cnt = sorted(cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    mask = cv2.drawContours(mask, [cnt],-1, 255, -1)
    return mask

def KMeans_mask(img):
    fp_size = [img.shape[0],img.shape[1],3]
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10
    
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((fp_size))
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)

    final = np.array(result_image, dtype = np.uint8)
    final = np.where(final > 128, 0, 255)
    final = final.astype(np.uint8)
    final = cv2.dilate(final, None, iterations=5)
    edge = cv2.morphologyEx(final,cv2.MORPH_OPEN,None, iterations=5)

    edge = np.array(edge)
    edge = np.where(edge > 50, 255, 0)
    edge = edge.astype(np.uint8)
    cnt = sorted(cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    mask = cv2.drawContours(mask, [cnt],-1, 255, -1)
    return mask

def image_enhance(img):
    
    # normalise the image and find a ROI
    blksze = 16
    thresh = 0.1 #modified from 0.1
    normim,mask = ridge_segment(img,blksze,thresh)
    #img = segment(img);
    
    _, mask2 = ridge_segment(img, blksze, 0.2)
    mask_sobel = Sobel_mask(img)
    mask_canny = Canny_mask(img)
    mask_kmeans = KMeans_mask(img)
    
    masks = [mask, mask2, mask_sobel, mask_canny, mask_kmeans]
    
    #mean_val = np.mean(img[mask])
    #std_val = np.std(img[mask])
    #normim = (img - mean_val)/(std_val)
    
    # find orientation of every pixel using gradients (smoothed)
    gradientsigma = 1
    blocksigma = 3
    orientsmoothsigma = 3
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma);

    #find the overall frequency of ridges
    blksze = 16 #modified from 32
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freqim,medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,maxWaveLength);

    # create gabor filter and do the actual filtering
    freq = medfreq*mask
    kx = 0.65;ky = 0.65
    enhim = ridge_filter(normim, orientim, freq, kx, ky)


    #th, bin_im = cv2.threshold(np.uint8(newim),0,255,cv2.THRESH_BINARY);
#    return(newim < -3)
    return(enhim, masks, orientim, freqim)
