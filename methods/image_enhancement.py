import cv2
import numpy as np


def sift_creator(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("The image could not be loaded.")
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    img_sift = cv2.drawKeypoints(image, keypoints, None)

    return img_sift, descriptors

def extract_features(images):
    descriptors_list = []

    for img in images:
        _, descriptors = sift_creator(image=img)
        if descriptors is not None:
            descriptors_list.append(descriptors.mean(axis=0))
        else:
            descriptors_list.append(np.zeros(128))
    
    return np.array(descriptors_list)

# Image enhancement for low level vision: image inpainting
# Image Inpainting is a task of reconstructing missing regions in an image.
def image_inpainting(image_path=None, image=None, inpaint_radius=3, block_size=9, C=7):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("The image could not be loaded.")

    mask = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, C)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    inpainted_image = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    return inpainted_image

# Image enhancement for low level vision: image resizing
def image_resizing(image_path=None, image=None, dimensions=(48,48), interpolation=cv2.INTER_LINEAR):
    # image: the image to resize.
    # dimensions: the new image dimensions as a tuple (width, height).
    # interpolation: method of interpolation.
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    if image is None:
        raise ValueError("The image could not be loaded.")
    
    resized_image = cv2.resize(image, dimensions, interpolation=interpolation)
    return resized_image

# Histogram equalization to enhance features of images
def histogram_equalization(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("The image could not be loaded.")
    
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Reduce noise by median blur
def reduce_noise_median(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    if image is None:
        raise ValueError("The image could not be loaded.")
    
    denoised_image = cv2.medianBlur(image, 3)
    return denoised_image

# Reduce noise by gaussian filter
def reduce_noise_gaussian(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("The image could not be loaded.")
    
    denoised_image = cv2.GaussianBlur(image, (3, 3), 0)
    return denoised_image

# Reduce noise: firstly, gaussian filter secondly, median blur
def smooth_image_first_gaussian(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    if image is None:
        raise ValueError("The image could not be loaded.")
    
    image = reduce_noise_gaussian(image=image)
    image = reduce_noise_median(image=image)
    return image

# Reduce noise: firstly, median blur secondly, Gaussian filter
def smooth_image_first_median(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
    if image is None:
        raise ValueError("The image could not be loaded.")
    
    image = reduce_noise_median(image=image)
    image = reduce_noise_gaussian(image=image)
    return image