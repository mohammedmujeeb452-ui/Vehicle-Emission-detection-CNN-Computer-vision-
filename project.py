import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

DATASET_PATH = "E:/dataset"  # here we have to set the location of dataset

img_height, img_width = 224, 224
batch_size = 32

# Load the dataset from directory structure
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names
class_names = train_ds.class_names
print("Classes:", class_names)

# Display a 3x3 grid of sample images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Normalize images
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Count number of images per class
count = {name: 0 for name in class_names}
for _, labels in train_ds:
    for lbl in labels.numpy():
        count[class_names[lbl]] += 1
print("Class distribution:", count)

# Plot class distribution bar chart
plt.bar(count.keys(), count.values(), color=['green','red'])
plt.title("Class Distribution (Smoke vs No Smoke)")
plt.show()


smoke_dir = os.path.join(DATASET_PATH, "smoke")
sample_files = [f for f in os.listdir(smoke_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
if len(sample_files) == 0:
    raise FileNotFoundError("No images found in the 'smoke' folder. Check your dataset path and folder names.")
sample_img_path = os.path.join(smoke_dir, sample_files)
img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
plt.hist(img.ravel(), bins=256, color='blue', alpha=0.7)
plt.title("Pixel Intensity Distribution - Smoke Image")
plt.show()
