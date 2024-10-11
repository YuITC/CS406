import cv2

def denoise_image(image, k=5):
    mean_denoise   = cv2.blur(image, (k, k))
    gauss_denoise  = cv2.GaussianBlur(image, (k, k), 0)
    median_denoise = cv2.medianBlur(image, k)

    images = [mean_denoise, gauss_denoise, median_denoise]
    titles = ['Mean Denoise', 'Gauss Denoise', 'Median Denoise']
    return images, titles