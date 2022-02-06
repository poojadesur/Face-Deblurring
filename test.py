
import os
import sys
import cv2
import numpy as np

# def snr(y, x):
#     signal = np.var(x)
#     noise = np.mean((y-x)**2)
#     return 10*np.log10(signal / noise)

image = cv2.imread('./outputs/Salient_Edges.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray_image = 3 * gray_image
cv2.imwrite("./outputs/edges.jpg",gray_image)

# # inter1 = cv2.imread('./outputs/blur_image.png')
# inter1 = cv2.imread('./outputs/intermediate_latent_image1.png')
# inter1 = cv2.cvtColor(inter1, cv2.COLOR_BGR2RGB)
# inter1 = cv2.cvtColor(inter1, cv2.COLOR_RGB2GRAY)

# inter7 = cv2.imread('./outputs/intermediate_latent_image19.png')
# inter7 = cv2.cvtColor(inter7, cv2.COLOR_BGR2RGB)
# inter7 = cv2.cvtColor(inter7, cv2.COLOR_RGB2GRAY)


# # blur_inter7 = inter7 - gray_image
# inter1_inter7 = inter7 - inter1

# print(np.unique(inter1_inter7))

# cv2.imwrite('./outputs/inter1_inter7.png',inter1_inter7)
# # cv2.imwrite('./outputs/inter1_inter7.png',inter1_inter7)

# print(snr(inter1,inter7))


# # 20.100545694500042





