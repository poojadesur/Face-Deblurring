import cv2
import numpy as np
import scipy
from scipy import signal

def flip_matrix(matrix):
    new_matrix = np.flip(matrix,1)
    new_matrix = np.flip(new_matrix,0)
    return new_matrix

def convolve(matrix_a, matrix_b , mode='valid'):
    return  signal.convolve2d(matrix_a,matrix_b,mode=mode)

def Prewitt(image, direction=None, mode='valid'):

    image = image.astype('float64')
    kernelx = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])

    kernely = np.array([[1, 1, 1],
                        [0, 0, 0],
                        [-1,-1,-1]])

    grad_x = convolve(image,flip_matrix(kernelx),mode)
    grad_y = convolve(image,flip_matrix(kernely),mode)
    if (direction == 'x'): return grad_x
        
    if (direction == 'y'): return grad_y
    
    return np.sqrt((grad_x ** 2) + (grad_y ** 2)).astype("uint8")























# def convolute(A, k):
#     new_h = A.shape[0] - k.shape[0] + 1
#     new_w = A.shape[1] - k.shape[1] + 1
    
#     A_out = np.zeros(shape = (new_h, new_w))
    

#     for i in range(new_h):
#         for j in range(new_w):
#             # for c in range(3):
#             A_out[i,j] = np.sum(A[i:i + k.shape[0],j:j + k.shape[1]] * k)

#     print(A_out.shape, A.shape)      
#     return A_out


























# def Sobeltest(image , ksize):
    
#     gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
#     cv2.imwrite('../outputs/test_x2.jpg' , gX)

#     gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
#     cv2.imwrite('../outputs/test_y2.jpg' , gY)
    
#     gX = cv2.convertScaleAbs(gX)
#     gY = cv2.convertScaleAbs(gY)
#     combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
#     return combined





































# test_image = cv2.imread('/home/manasvi/3rdYearSem5/DIP/dip-project-newt/exemplar_dataset/0.png')
# test_image = cv2.cvtColor(test_image,cv2.COLOR_RGB2GRAY)
# testx = Prewitt(test_image , direction='x')
# testy = Prewitt(test_image,direction="y")
# testxy = Prewitt(test_image)

# cv2.imwrite('../outputs/test_x.jpg' , testx)
# cv2.imwrite('../outputs/testy.jpg',testy)
# cv2.imwrite('../outputs/testxy.jpg',testxy)








# def apply_kernel_single(img, kernel, new_channel_shape):
#     new_img = np.zeros(new_channel_shape, np.int)
#     for i in range(new_channel_shape[0]):
#         for j in range(new_channel_shape[1]):
#             new_img[i][j] = np.round(np.sum(img[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel))
#     return new_img

# def apply_kernel(img, kernel, zero_padding = 0):
#     in_img = np.zeros((img.shape[0] + 2 * zero_padding, img.shape[1] + 2 * zero_padding), int)
#     in_img[zero_padding : in_img.shape[0] - zero_padding, zero_padding : in_img.shape[1] - zero_padding] = img
    
#     new_h, new_w = in_img.shape[0] - kernel.shape[0] + 1, in_img.shape[1] - kernel.shape[1] + 1
    
#     if len(img.shape) == 2:
#         return apply_kernel_single(in_img, kernel, (new_h, new_w))
        
#     new_img = np.zeros((new_h, new_w, img.shape[2]), np.int)
#     for c in range(img.shape[2]):
#         new_img[:,:,c] = apply_kernel_single(in_img[:,:,c], kernel, (new_h, new_w))
#     return new_img

# def sobel_filter(img, direction = 'x', zero_padding = 0):
#     if (direction == 'x'):
#         Mx = np.array([[-1, 0, 1],
#                        [-2, 0, 2],
#                        [-1, 0, 1]])
#         return apply_kernel(img, Mx, zero_padding).astype("uint8")
#     if (direction == 'y'):
#         My = np.array([[1, 2, 1],
#                        [0, 0, 0],
#                        [-1,-2,-1]])
#         return apply_kernel(img, My, zero_padding).astype("uint8")
    
#     Mx = np.array([[-1, 0, 1],
#                    [-2, 0, 2],
#                    [-1, 0, 1]])
#     My = np.array([[1, 2, 1],
#                    [0, 0, 0],
#                    [-1,-2,-1]])
#     return np.sqrt(apply_kernel(img, Mx,zero_padding) ** 2 + apply_kernel(img, My, zero_padding) ** 2).astype("uint8")

