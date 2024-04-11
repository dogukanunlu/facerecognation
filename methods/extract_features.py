from skimage.feature import local_binary_pattern, hog
import numpy as np
import cv2

# return sift descriptors for each image
def sift_creator(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError("The image could not be loaded.")

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    img_sift = cv2.drawKeypoints(image, keypoints, None)

    return img_sift, descriptors

# return sift descriptors list for all images
def extract_features_sift(images):
    descriptors_list = []

    for img in images:
        _, descriptors = sift_creator(image=img)
        if descriptors is not None:
            descriptors_list.append(descriptors.mean(axis=0))
        else:
            descriptors_list.append(np.zeros(128))
    
    return descriptors_list


def extract_features_lbp(img=None, images=None):
    features = []
    
    if img is not None:
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp, density=True, bins=np.arange(257), range=(0, 256))
        return hist
    
    for image in images:
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp, density=True, bins=np.arange(257), range=(0, 256))
        features.append(hist)
    
    return np.array(features)

def extract_features_hog(img=None, images=None):
    features = []
    
    if img is not None:
        return hog(img, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=False, multichannel=False)
    
    for image in images:
        hog_features = hog(image, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=False, multichannel=False)
        features.append(hog_features)
        
    return np.array(features)

def extract_combined_features(images):
    """
    Extract combined HOG and LBP features from a list of images.
    """
    combined_features = []
    for image in images:
        # Extract HOG features
        hog_features = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, multichannel=False)

        # Extract LBP features
        lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, density=True, bins=np.arange(257), range=(0, 256))
        
        # Combine HOG and LBP features
        combined = np.hstack((hog_features, lbp_hist))
        combined_features.append(combined)
    
    return np.array(combined_features)