import streamlit as st
import cv2
import numpy as np
from PIL import Image
from denoise_image import denoise_image
from sharpen_image import sharpen_image
from edge_detection import edge_detection
from utils import plot_image, create_noise_image, create_blur_image

def main():
    st.set_page_config(layout='wide')
    st.title('Digital Image Processing Webapp')
    st.markdown(
        '''
        This app provides the following image processing functionalities:
        1. **Denoising / Smoothing** using Mean, Gaussian, and Median methods.
        2. **Sharpening** using Laplacian, Unsharp Masking, and Custom Sharpening Kernel methods.
        3. **Edge Detection** using Sobel, Prewitt, and Canny methods.
        '''
    )
    st.divider()
    base_img, noise_img, blur_img = None, None, None

    # -------------------------------------------------------------------------------------------------------------------------------
    st.header('Please upload your image!')
    col1, col2 = st.columns(2)
    with col1:
        opt = st.selectbox('Choose upload type:', ('Use your image', 'Use demo image'))
    with col2:
        if opt == 'Use your image':
            upload   = st.file_uploader('Choose your image:', type=['jpg', 'jpeg', 'png'])
            base_img = np.array(Image.open(upload)) if upload is not None else base_img
        else:
            base_img = np.array(Image.open('data/cecilia-color.jpg'))

    # -------------------------------------------------------------------------------------------------------------------------------
    if base_img is not None:
        with col2:
            with st.columns([1, 3, 1])[1]:
                st.pyplot(plot_image([base_img], ['Your Image' if opt == 'Use your image' else 'Demo Image']))

        # ---------------------------------------------------------------------------------------------------------------------------
        st.header('1. Denoising / Smoothing Image')
        noise_images, noise_titles = create_noise_image(base_img)

        option = ['Gaussian Noise', 'Salt and Pepper Noise']
        noise_opt = st.selectbox('Choose noise type:', option)
        noise_idx = option.index(noise_opt)
        noise_img, noise_tit = noise_images[noise_idx], noise_titles[noise_idx]

        denoise_images, denoise_titles = denoise_image(noise_img)
        st.pyplot(plot_image([noise_img] + denoise_images, [noise_tit] + denoise_titles))

        # ---------------------------------------------------------------------------------------------------------------------------
        st.header('2. Sharpening Image')
        blur_images, blur_titles = create_blur_image(base_img)

        option = ['Mean Blur', 'Gauss Blur', 'Median Blur']
        blur_opt = st.selectbox('Choose blur type:', option)
        blur_idx = option.index(blur_opt)
        blur_img, blur_tit = blur_images[blur_idx], blur_titles[blur_idx]

        sharpen_images, sharpen_titles = sharpen_image(blur_img)
        st.pyplot(plot_image([blur_img] + sharpen_images, [blur_tit] + sharpen_titles))

        # ---------------------------------------------------------------------------------------------------------------------------
        st.header('3. Edge Detection')

        edge_images, edge_titles = edge_detection(base_img)
        gray_img = cv2.cvtColor(base_img, cv2.COLOR_RGB2GRAY)   
        st.pyplot(plot_image([gray_img] + edge_images, ['Grayscale Image'] + edge_titles))

    
if __name__ == "__main__":
    main()