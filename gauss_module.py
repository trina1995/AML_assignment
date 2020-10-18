#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2


# """
# Gaussian function taking as argument the standard deviation sigma
# The filter should be defined for all integer values x in the range [-3sigma,3sigma]
# The function should return the Gaussian values Gx computed at the indexes x
# """

# In[2]:


def gauss(sigma):
    
    x = np.arange (-3*sigma, 3*sigma, 1)
    Gx=(1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x**2)/(2*sigma**2)))
    
    return Gx, x


# """
# Implement a 2D Gaussian filter, leveraging the previous gauss.
# Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
# Leverage the separability of Gaussian filtering
# Input: image, sigma (standard deviation)
# Output: smoothed image
# """

# In[ ]:


def gaussianfilter(img, sigma):
    
    Gx, x = gauss(sigma)

    horizontal_kernel = Gx.reshape((1, -1))
    vertical_kernel = Gx.reshape((-1, 1))

    horizontal_conv = conv2(img, horizontal_kernel, mode='same', boundary='fill')
    vertical_conv = conv2(horizontal_conv, vertical_kernel, mode='same', boundary='fill')

    smooth_img = vertical_conv
    
    return smooth_img


# 
# """
# Gaussian derivative function taking as argument the standard deviation sigma
# The filter should be defined for all integer values x in the range [-3sigma,3sigma]
# The function should return the Gaussian derivative values Dx computed at the indexes x
# """

# In[3]:


def gaussdx(sigma):
    
    x = np.arange(-3*sigma, 3*sigma, 1)
    Dx = -1/(np.sqrt(2*np.pi)*sigma**3)*x*np.exp(-x**2/(2*sigma**2))
    
    return Dx, x


# In[4]:


def gaussderiv(img, sigma):
    
    dx, x = gaussdx(sigma)

    horizontal_kernel = dx.reshape((1, -1))
    vertical_kernel = dx.reshape((-1, 1))

    img_Dx = conv2(img, horizontal_kernel, mode='same')
    img_Dy = conv2(img, vertical_kernel, mode='same')

    return img_Dx, img_Dy

