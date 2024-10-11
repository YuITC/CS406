import streamlit as st
import numpy as np
from PIL import Image
from denoise_image import denoise_image
from sharpen_image import sharpen_image
from edge_detection import edge_detection
from utils import plot_image, create_noise_image, create_blur_image

base_img = Image.open('test/data/cecilia-color.jpg')
base_img = np.array(base_img)

print(type(create_noise_image(base_img)))

opt = None

st.header('1. Denoising / Smoothing image')
img = base_img if opt == 'Use your image' else 
st.header('2. Sharpening image')

st.header('3. Edge Detection')


if opt == 'Use your image':
    st.header('1. Denoising / Smoothing image')
    denoise_images, denoise_titles = denoise_image(base_img)
    st.pyplot(plot_image([base_img] + denoise_images, ['Original Image'] + denoise_titles))

    st.header('2. Sharpening image')
    sharpen_images, sharpen_titles = sharpen_image(base_img)
    st.pyplot(plot_image([base_img] + sharpen_images, ['Your Image'] + sharpen_titles))

    st.header('3. Edge Detection image')
    edge_images, edge_titles = edge_detection(base_img)
    st.pyplot(plot_image([base_img] + edge_images, ['Your Image'] + edge_titles))
else:
    noise_img = create_noise_image(base_img)[0][-1]
    blur_img  = create_blur_image(base_img)[0][0]

    st.header('1. Denoising / Smoothing image')
    denoise_images, denoise_titles = denoise_image(noise_img)
    st.pyplot(plot_image([noise_img] + denoise_images, ['Demo Nosie Image (Salt and pepper Noise)'] + denoise_titles))

    st.header('2. Sharpening image')
    sharpen_images, sharpen_titles = sharpen_image(blur_img)
    st.pyplot(plot_image([blur_img] + sharpen_images, ['Demo Blur Image (Mean Blur)'] + sharpen_titles))

    st.header('3. Edge Detection')
    edge_images, edge_titles = edge_detection(base_img)
    st.pyplot(plot_image([base_img] + edge_images, ['Demo Base Image'] + edge_titles))