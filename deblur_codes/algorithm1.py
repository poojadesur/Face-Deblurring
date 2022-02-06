import numpy as np
from  utils.gradient import *
from deblur_codes.algorithm1 import *
import math
from tqdm import tqdm
from deblur_codes.deconvolution import *

''' argue about how to get wx later '''

def get_w(I, lambda_, beta):

    delI = Prewitt(I,mode="same")
    delI_x = Prewitt(I,direction="x",mode="same")
    delI_y = Prewitt(I,direction="y",mode="same")

    if np.square(np.linalg.norm(delI, ord = 1)) >= lambda_/beta:
        w_x = delI_x
        w_y = delI_y
        return w_x,w_y
    
    return np.zeros(delI_x.shape),np.zeros(delI_y.shape)

def pad_to_size(image , szx, szy):
    
    width = image.shape[1]
    height = image.shape[0]
    left = math.ceil((szx - width)/2)
    right = math.floor((szx - width)/2)

    top = math.ceil((szy - height)/2)
    bottom = math.floor((szy - height)/2)
    
    padded_img = np.pad(image , ((top , bottom) , (left , right)) , 'constant')
    return padded_img

def get_I(blur_image, blur_kernel, w_x, w_y, beta):
    
    # dx = np.array(  [[-1, 0, 1],
    #                 [-1, 0, 1],
    #                 [-1, 0, 1]])

    # dy = np.array(  [[1, 1, 1],
    #                 [0, 0, 0],
    #                 [-1,-1,-1]])

    # B_ft = np.fft.fftshift(np.fft.fft2(blur_image , axes=(0,1)))

    # cv2.imwrite('./outputs/blur_image_fft.png', blur_image)  

    # padded_blur_kernel = pad_to_size(blur_kernel, blur_image.shape[1], blur_image.shape[0])
    # k_ft = np.fft.fftshift(np.fft.fft2(padded_blur_kernel, axes=(0,1)))
    # k_ft_c = np.conjugate(k_ft)
    
    # padded_dx = pad_to_size(dx, blur_image.shape[1], blur_image.shape[0])
    # dx_ft = np.fft.fftshift(np.fft.fft2(padded_dx, axes=(0,1)))
    # dx_ft_c = np.conjugate(dx_ft)
    
    # padded_dy = pad_to_size(dy, blur_image.shape[1], blur_image.shape[0])
    # dy_ft = np.fft.fftshift(np.fft.fft2(padded_dy, axes=(0,1)))
    # dy_ft_c = np.conjugate(dy_ft)
    
    # wx_ft = np.fft.fftshift(np.fft.fft2(w_x, axes=(0,1)))
    # wy_ft = np.fft.fftshift(np.fft.fft2(w_y, axes=(0,1)))
    
    # I_ft = np.divide((k_ft_c * B_ft + beta * (dx_ft_c * wx_ft + dy_ft_c * wy_ft))
    #                  ,(k_ft_c * k_ft + beta * (dx_ft_c * dx_ft + dy_ft_c * dy_ft)))
    
    # I_ft = np.divide(( B_ft )
    #                  ,(k_ft))
    
    # I_ft = np.fft.ifftshift(I_ft)
                     
    # return np.fft.ifft2(I_ft).real

    I = gd_deconvolution(blur_image, blur_kernel, 100, 1e-4)

    return I

def algorithm1(blur_image, blur_kernel , beta=4e-3 , lambda_=2e-3 , max_beta=12e-3):
    
    # I = np.copy(blur_image)
    # print("Algorithm1 initial beta",beta)

    I = gd_deconvolution(blur_image, blur_kernel, 100, 1e-4)

    # while(beta <= max_beta):
    #     w_x,w_y = get_w(I,lambda_,beta)
    #     I = get_I(blur_image,blur_kernel, w_x, w_y, beta)
    #     beta = 2*beta
    
    # print("Algorithm1 final beta",beta)

    return I



