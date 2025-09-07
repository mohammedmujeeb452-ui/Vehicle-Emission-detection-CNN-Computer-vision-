# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import cv2

# DATASET_PATH = "E:/dataset"  # here we have to set the location of dataset

# img_height, img_width = 224, 224
# batch_size = 32

# # Load the dataset from directory structure
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_PATH,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )

# # Get class names
# class_names = train_ds.class_names
# print("Classes:", class_names)

# # Display a 3x3 grid of sample images
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
# plt.show()

# # Normalize images
# normalization_layer = tf.keras.layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# # Count number of images per class
# count = {name: 0 for name in class_names}
# for _, labels in train_ds:
#     for lbl in labels.numpy():
#         count[class_names[lbl]] += 1
# print("Class distribution:", count)

# # Plot class distribution bar chart
# plt.bar(count.keys(), count.values(), color=['green','red'])
# plt.title("Class Distribution (Smoke vs No Smoke)")
# plt.show()


# smoke_dir = os.path.join(DATASET_PATH, "smoke")
# sample_files = [f for f in os.listdir(smoke_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
# if len(sample_files) == 0:
#     raise FileNotFoundError("No images found in the 'smoke' folder. Check your dataset path and folder names.")
# sample_img_path = os.path.join(smoke_dir, sample_files)
# img = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
# plt.hist(img.ravel(), bins=256, color='blue', alpha=0.7)
# plt.title("Pixel Intensity Distribution - Smoke Image")
# plt.show()


import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Set dataset path and parameters
DATASET_PATH = "E:/dataset"  # Change to your actual dataset folder
img_height, img_width = 224, 224
batch_size = 32

# Load dataset from directory
dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

# Gather all images and labels into numpy arrays
images = []
labels = []
for batch_images, batch_labels in dataset:
    images.extend(batch_images.numpy())
    labels.extend(batch_labels.numpy())

images = np.array(images)
labels = np.array(labels)

# Split dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)
print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

# Normalize image data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the CNN model
from tensorflow.keras import layers, models
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(dataset.class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train, epochs=10, validation_split=0.2
)

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Plot training and validation accuracy curves
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
