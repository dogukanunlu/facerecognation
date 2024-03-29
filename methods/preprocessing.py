from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

def load_dataset(dataset_path):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        for file_name in os.listdir(os.path.join(dataset_path, label)):
            img = cv2.imread(os.path.join(dataset_path, label, file_name), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    
    return images, labels

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