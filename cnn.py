import os
import cv2
import imghdr
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import image_dataset_from_directory
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    class_path = os.path.join(data_dir, image_class)
    for image in os.listdir(class_path):
        image_path = os.path.join(class_path, image)
        try:
            img = cv2.imread(image_path)
            typ = imghdr.what(image_path)
            if typ not in image_exts:
                print(f"Removed invalid image: {image_path}")
                os.remove(image_path)
        except Exception as e:
            print(f"Issue with image {image_path}: {e}")
dataset = image_dataset_from_directory('data', image_size=(256, 256), batch_size=32)
class_names = dataset.class_names
print("Classes:", class_names)

dataset = dataset.map(lambda x, y: (x / 255.0, y))
data_size = dataset.cardinality().numpy()
train_size = int(data_size * 0.6)
val_size = int(data_size * 0.3)

train = dataset.take(train_size)
val = dataset.skip(train_size).take(val_size)
test = dataset.skip(train_size + val_size)

data_iter = dataset.as_numpy_iterator()
images, labels = next(data_iter)

fig, ax = plt.subplots(5, 5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        ax[i][j].imshow(images[i * 5 + j])
        ax[i][j].axis('off')
plt.show()

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']
)

history = model.fit(train, validation_data=val, epochs=10)
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Accuracy: {test_accuracy:.2f}")
