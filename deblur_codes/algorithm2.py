import numpy as np
from utils.gradient import *
from deblur_codes.algorithm1 import *
from tqdm import tqdm
from deblur_codes.deconv import *

def estimate_kernel(del_B, del_S, gamma = 1, lr = 0.1, epochs = 10, tol = 1e-10):

    k = np.random.rand(3,3)
    k = k/np.sum(k)
    
    print("Estimate kernels")

    for i in tqdm(range(epochs)):
        
        val1 = convolve(del_S, flip_matrix(k))      # 1024 - 1022
        val2 = val1 - del_B                         # 1022 - 1022
        val3 = convolve(del_S, val2)                # 1022 - 3

        del_k = 2 * (val3 + gamma*k)
        
        if (del_k < tol).all():
            break

        k = k - lr * del_k
        k = k/np.sum(k)
        
    return k

def algorithm2(blur_image, delta_S, n_iters):
    
    I = np.zeros(shape=(500,500))
    print("Iterations")
    cv2.imwrite('./outputs/blur_image.png', blur_image)  

    kernel_size = 30
    kernel_v = np.zeros((kernel_size, kernel_size))

    kernel_h = np.copy(kernel_v)

    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)

    kernel_v /= kernel_size
    kernel_h /= kernel_size

    kernel_h += np.random.exponential(scale=1e-4)
    kernel_h /= np.sum(kernel_h)

    blur_kernel = kernel_h
    lam = 1e6  # regularization parameter, tuned to maximize SNR
    I = deconv(blur_image, blur_kernel, lam, beta_max=256)

    # for l in tqdm(range(n_iters)):

    #     cv2.imwrite('./outputs/intermediate_latent_image' + str(l) + '.png', I)  
    #     delta_B = Prewitt(blur_image,mode="valid")
    #     blur_kernel = estimate_kernel(delta_B, delta_S , epochs=100)
    #     I = algorithm1(blur_image,blur_kernel)
    #     delta_S = Prewitt(I,mode="same")

    return blur_kernel, I


