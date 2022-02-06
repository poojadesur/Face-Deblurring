
import os
import sys
import cv2
from deblur_codes.getBestExemplar import *
from deblur_codes.getSalientEdges import *
from deblur_codes.algorithm2 import *
from deblur_codes.algorithm1 import *
from deblur_codes.deconvolution import *



def saveOutputs(image, image_path):
    
    try:
        os.mkdir('outputs')
    except FileExistsError:
        pass

    cv2.imwrite(image_path, image)    


if __name__ == "__main__":
 
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    """ 
        Step 1: Get Best matching exemplar 
    """

    exemplar = get_best_exemplar(gray_image, "./exemplar_dataset")
    gray_exemplar = cv2.cvtColor(exemplar, cv2.COLOR_RGB2GRAY)
    saveOutputs(exemplar,"./outputs/Best_Exemplar.png")

    """ 
        Step 2: Get Salient edges of the best matched exemplar image
    """
    
    mask , salient_edges = get_salient_edges(gray_exemplar)
    print(salient_edges.shape)

    saveOutputs(mask,"./outputs/Mask.png")
    saveOutputs(salient_edges,"./outputs/Salient_Edges.png")

    """ 
        Step 3: Predict the best Blur kernel
    """

    blur_kernel,I = algorithm2(gray_image , salient_edges , n_iters = 2)
    print(blur_kernel)
    saveOutputs(I, "./outputs/Intermediate_Latent_Image.png")

    print("Finished Step 3")

    # lambda_ = 0.002
    # beta = 0.004

    # print(gray_image.shape)
    
    # blur_kernel = np.random.rand(3,3)
    # blur_kernel = blur_kernel/np.sum(blur_kernel)

    # w_x,w_y = get_w(gray_image,lambda_,beta)

    # I = get_I(gray_image,k, w_x, w_y, beta)
    # cv2.imwrite('./outputs/test.png',I)


    """
        Step 4: Recover Blurred image using non blind deconvulution filter
    """ 
    
    # deblurred_image = gd_deconvolution(gray_image, blur_kernel, epochs=200, lr=1e-4, tol = 1e-10)
    # saveOutputs(deblurred_image, "./outputs/Deblurred_Image.png")
    
