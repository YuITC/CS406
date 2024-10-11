import cv2
import numpy as np

def sharpen_image(image):
    # if len(image.shape) == 3:
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_image)
    
    laplacian = cv2.Laplacian(y, cv2.CV_64F)
    laplacian_edges = cv2.convertScaleAbs(laplacian)
    enhanced_y = cv2.addWeighted(y, 1.0, laplacian_edges, 0.5, 0)
    
    laplacian_image = cv2.merge((enhanced_y, u, v))
    laplacian_image = cv2.cvtColor(laplacian_image, cv2.COLOR_YUV2BGR)
    # else:
    #     laplacian = cv2.Laplacian(image, cv2.CV_64F)
    #     laplacian_edges = cv2.convertScaleAbs(laplacian)
    #     laplacian_image = cv2.addWeighted(image, 1.0, laplacian_edges, 0.5, 0)
    
    gaussian_blur = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    custom_image = cv2.filter2D(image, -1, kernel)

    images = [laplacian_image, unsharp_image, custom_image]
    titles = ['Laplacian Sharpened', 'Unsharp Masked', 'Custom kernel Sharpened']
    return images, titles