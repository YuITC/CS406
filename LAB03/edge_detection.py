import cv2
import numpy as np

def edge_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    sobel_x         = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y         = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges     = cv2.convertScaleAbs(sobel_magnitude)

    prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x        = cv2.filter2D(image, -1, prewitt_kernel_x)
    prewitt_y        = cv2.filter2D(image, -1, prewitt_kernel_y)
    prewitt_edges    = cv2.convertScaleAbs(prewitt_x + prewitt_y)

    canny_edges = cv2.Canny(image, 100, 200)

    images = [sobel_edges, prewitt_edges, canny_edges]
    titles = ['Sobel Edge Detection', 'Prewitt Edge Detection', 'Canny Edge Detection']
    return images, titles