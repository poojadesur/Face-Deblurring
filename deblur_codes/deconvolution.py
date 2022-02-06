import numpy as np
from utils.gradient import convolve, flip_matrix
import cv2
from tqdm import tqdm

def loss(I, k, B):
    return np.linalg.norm(convolve(I, k, "same") - B)

def gd_deconvolution(B, k, epochs, lr, tol = 1e-10):
    
    # I = np.random.randint(0,255,(1024,1024))

    I = B.copy()
    q = k.T
    print("****************Deconvolution Gradient Descent***************")

    errors = []
    for i in tqdm(range(epochs)):        
        # print(f"Epoch: {epoch}")
        val1 = convolve(I, flip_matrix(k), "same")
        val2 = val1 - B
        del_I = convolve(val2, flip_matrix(q), "same")
        
        if (del_I < tol).all():
            break

        I = I - lr * del_I
        
        errors.append(loss(I, flip_matrix(k), B))

    # plt.plot(np.arange(len(errors)), errors)
    # plt.savefig("loss.jpg")
        
    return I.astype('uint8')


