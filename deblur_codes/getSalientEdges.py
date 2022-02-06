from utils.featureDetection import *
from utils.gradient import *

def get_salient_edges(image):

    mask = feature_Detection(image)
    gradient_image = Prewitt(image,mode="same")

    cv2.imwrite("./outputs/edges.png",gradient_image)

    final_image = np.zeros(image.shape)
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if mask[row][col] != 0:
                final_image[row][col] = gradient_image[row][col]

    return mask, final_image