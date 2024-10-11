import cv2
import matplotlib.pyplot as plt
import numpy as np

### Create blur image
def create_blur_image(image, k=5):
    mean_blur   = cv2.blur(image, (k, k))
    gauss_blur  = cv2.GaussianBlur(image, (k, k), 0)
    median_blur = cv2.medianBlur(image, k)

    images = [mean_blur, gauss_blur, median_blur]
    titles = ['Mean Blur', 'Gauss Blur', 'Median Blur']
    return images, titles

### Create noise image
def create_noise_image(image):
    def add_gauss_noise(image, mean=0, sigma=25):
        return np.clip(image + np.random.normal(mean, sigma, image.shape), 0, 255).astype(np.uint8)
    
    def add_SAP_noise(image, salt_prob=0.01, pepper_prob=0.01):
        def apply_noise(image, prob, value):
            noised_image = np.copy(image)
            h, w = image.shape[:2]
            num_noise = np.ceil(prob * h * w).astype(int)
            coords = [np.random.randint(0, i, num_noise) for i in (h, w)]
            
            noised_image[coords[0], coords[1]] = value if len(image.shape) == 2 else [value] * 3
            return noised_image
    
        noised_image = apply_noise(image, salt_prob, 255)
        noised_image = apply_noise(noised_image, pepper_prob, 0)
    
        return noised_image

    gaussian_noise    = add_gauss_noise(image)
    salt_pepper_noise = add_SAP_noise(image)

    images = [gaussian_noise, salt_pepper_noise]
    titles = ['Gaussian Noise', 'Salt and Pepper Noise']
    return images, titles

### Plot images
def plot_image(images, titles=[], images_per_row=4):
    n = len(images)
    c = min(images_per_row, n)
    r = int(np.ceil(n / c))

    if len(titles) < n:
        titles.extend([''] * (n - len(titles)))

    fig, axes = plt.subplots(r, c, figsize=(4 * c, 4 * r))

    for image, title, ax in zip(images, titles, axes.ravel() if n > 1 else [axes]):
        if image is None:
            ax.axis('off')
            continue
        ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    if n > 1:
        for ax in axes.ravel()[n:]:
            ax.axis('off')

    plt.tight_layout()
    
    return fig
