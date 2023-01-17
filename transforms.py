'''
Transformation functions to be applied to an img array
'''
import numpy as np
import cv2
# TRANSFORMATION FUNCTIONS: DEFAULT (NON-RANDOMIZED)

def func_neg(img_arr):
    neg = 255 - img_arr # neg = (L-1) - img
    return neg

def func_log(img_arr):
    # Apply log transform.
    c = 255/(np.log(1 + np.max(img_arr)))
    log_transformed = c * np.log(1 + img_arr) 

    # Specify the data type. 
    log_transformed = np.array(log_transformed, dtype = 'uint8') 
    return log_transformed

def func_gamma(img_arr):
    gamma = 0.5
    gamma_corrected = np.array(255*(img_arr / 255) ** gamma, dtype = 'uint8') 
    return gamma_corrected

def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 


def func_piecewise_linear(img_arr):
    r1 = 75
    s1 = 55
    r2 = 150
    s2 = 225

    # Vectorize the function to apply it to each value in the Numpy array. 
    pixelVal_vec = np.vectorize(pixelVal) 

    # Apply contrast stretching. 
    contrast_stretched = pixelVal_vec(img_arr, r1, s1, r2, s2) 
    contrast_stretched = np.array(contrast_stretched, dtype = 'uint8')
    return contrast_stretched

def func_identity(img_arr):
    return img_arr


def func_sobelx(img_arr):
    # remove noise
    img = cv2.GaussianBlur(img_arr,(3,3),0)

    # convolute with sobel kernel
    sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    return sobelx


def func_sobely(img_arr):
    # remove noise                                                                                                                            
    img = cv2.GaussianBlur(img_arr,(3,3),0)

    # convolute with sobel kernel                                                                                                             
    sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
    return sobely


# EXPERIMENTAL
def func_blur(img_arr):
    # remove noise                                                                                                                            
    img = cv2.GaussianBlur(img_arr,(3,3),0)

    blurred = cv2.blur(img, (7,7))
    return blurred
    

def func_gamma_neg(img_arr):
    gamma = 0.3
    gamma_corrected = 255 - np.array(255*(img_arr / 255) ** gamma, dtype = 'uint8') 
    return gamma_corrected

def func_exp(img_arr):
    a = 0.02
    b = 2.3
    out = np.array(b * np.exp(a * img_arr), dtype='uint8')
    return out

# TRANSFORMATION FUNCTIONS: RANDOMIZED
# follows notation from 'Fix the Project: Ideas' in oneNote
# bounds for uniform priors to sample each function's parameters from
def func_identity_randomized(img_arr, seed):
    b0 = 0
    delb = 20
    theta0 = np.pi / 4
    deltheta = np.pi / 8

    rng = np.random.default_rng(seed)
    b = rng.uniform(b0 - delb, b0 + delb)
    theta = rng.uniform(theta0 - deltheta, theta0 + deltheta)
    m = np.tan(theta)
 
    return np.array(m * img_arr + b, dtype = 'uint8')


def func_neg_randomized(img_arr, seed):
    b0 = 255
    delb = 20
    theta0 = -np.pi / 4
    deltheta = np.pi / 8

    rng = np.random.default_rng(seed)
    b = rng.uniform(b0 - delb, b0 + delb)
    theta = rng.uniform(theta0 - deltheta, theta0 + deltheta)
    m = np.tan(theta)
 
    return np.array(m * img_arr + b, dtype = 'uint8')

def func_log_randomized(img_arr, seed):
    # Apply log transform.
    ctilde = 255/(np.log(1 + np.max(img_arr)))
    a0 = 1.
    dela = 0.3

    rng = np.random.default_rng(seed)
    a = rng.uniform(a0 - dela, a0 + dela)
    c = a * ctilde

    log_transformed = c * np.log(1 + img_arr) 

    # Specify the data type. 
    log_transformed = np.array(log_transformed, dtype = 'uint8') 
    return log_transformed

def func_gamma_randomized(img_arr, seed):
    alpha0 = 0.
    delalpha = 5.

    rng = np.random.default_rng(seed)
    alpha = rng.uniform(alpha0 - delalpha, alpha0 + delalpha)
    gamma = np.power(2., alpha)

    gamma_corrected = np.array(255*(img_arr / 255) ** gamma, dtype = 'uint8') 
    return gamma_corrected

def pixelVal(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 


def func_piecewise_linear_randomized(img_arr, seed):
    r10 = 75
    s10 = 55
    r20 = 150
    s20 = 225
    delta = 20

    rng = np.random.default_rng(seed)
    r1 = rng.uniform(r10 - delta, r10 + delta)
    r2 = rng.uniform(r20 - delta, r20 + delta)
    s1 = rng.uniform(s10 - delta, s10 + delta)
    s2 = rng.uniform(s20 - delta, s20 + delta)

    # Vectorize the function to apply it to each value in the Numpy array. 
    pixelVal_vec = np.vectorize(pixelVal) 

    # Apply contrast stretching. 
    contrast_stretched = pixelVal_vec(img_arr, r1, s1, r2, s2) 
    contrast_stretched = np.array(contrast_stretched, dtype = 'uint8')
    return contrast_stretched

def func_sobelx_randomized(img_arr, seed):
    # remove noise
    img = cv2.GaussianBlur(img_arr,(3,3),0)

    # convolute with sobel kernel
    sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    return sobelx


def func_sobely_randomized(img_arr, seed):
    # remove noise                                                                                                                            
    img = cv2.GaussianBlur(img_arr,(3,3),0)

    # convolute with sobel kernel                                                                                                             
    sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
    return sobely
