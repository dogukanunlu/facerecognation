from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from PIL import Image

def load_dataset(data_dir, limit_category=('happy', 5000)):
    categories = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Assuming 'disgust' is already removed
    images = []
    labels = []
    
    for label, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        if os.path.exists(category_dir):  # Check if category directory exists
            loaded_images_count = 0  # Counter for images of the current category
            for file in sorted(os.listdir(category_dir)):  # Sort files to maintain order
                if limit_category and category == limit_category[0] and loaded_images_count >= limit_category[1]:
                    break  # Stop if the limit for this category is reached
                file_path = os.path.join(category_dir, file)
                image = Image.open(file_path)
                image = image.resize((48, 48))  # Resize images for uniformity
                image = np.array(image)
                images.append(image)
                labels.append(label)
                loaded_images_count += 1
    
    # Convert lists to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    return X, y

def create_data_gen():
    return ImageDataGenerator(
            rescale=1./255
        )
    
def plot_example_images(images_arr, labels_arr, label_map):
    fig, axes = plt.subplots(1, 5, figsize=(10, 10))
    axes = axes.flatten()
    for img, label, ax in zip(images_arr, labels_arr, axes):
        class_label = label_map[np.argmax(label)]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(class_label)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
# this function will gives us singularity for image showing
# using matplotlib for displaying
def show_image_as_plot(image_path=None, image=None):
    if image_path:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    #plt.imshow(image_rgb)
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.show()