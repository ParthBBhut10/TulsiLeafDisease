import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.layers import Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
import tensorflow.keras.layers as layers
import tensorflow.keras.applications.resnet50 as resnet50
#from google.colab import drive
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

data_directory = 'C:\FInal Year Code\Basil'
categories = os.listdir(data_directory)
label_encoder = LabelEncoder()
label_encoder.fit(categories)

# Load the images
images = []
labels = []

for category in categories:
    path = os.path.join(data_directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)

        # Check if the image loaded successfully
        if image is not None:
            image = cv2.resize(image, (100, 100))
            images.append(image)
            labels.append(category)

# Convert the images to a NumPy array
images = np.array(images)
labels = np.array(labels)

# Encode the labels
labels = label_encoder.transform(labels)

np.save('label_encoder_classes.npy', label_encoder.classes_)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

# View the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap="gray")
    plt.xlabel(label_encoder.inverse_transform([y_train[i]]))
plt.show()

# Blur all images in the training and testing sets
X_train_blurred = []
X_test_blurred = []

for image in X_train:
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    X_train_blurred.append(blurred_image)

for image in X_test:
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    X_test_blurred.append(blurred_image)

# Convert the lists to NumPy arrays
X_train_blurred = np.array(X_train_blurred)
X_test_blurred = np.array(X_test_blurred)

# rgb to hsv conversion of the all the images

X_train_hsv = []
X_test_hsv = []

for image in X_train_blurred:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    X_train_hsv.append(hsv_image)

for image in X_test_blurred:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    X_test_hsv.append(hsv_image)

# Convert the lists to NumPy arrays
X_train_hsv = np.array(X_train_hsv)
X_test_hsv = np.array(X_test_hsv)


# Define the color threshold
lower_bound = np.array([0, 50, 50])
upper_bound = np.array([10, 255, 255])

# Threshold the images in the training and testing sets
X_train_thresholded = []
X_test_thresholded = []

for image in X_train_hsv:
    mask = cv2.inRange(image, lower_bound, upper_bound)
    thresholded_image = cv2.bitwise_and(image, image, mask=mask)
    X_train_thresholded.append(thresholded_image)

for image in X_test_hsv:
    mask = cv2.inRange(image, lower_bound, upper_bound)
    thresholded_image = cv2.bitwise_and(image, image, mask=mask)
    X_test_thresholded.append(thresholded_image)

# Convert the lists to NumPy arrays
X_train_thresholded = np.array(X_train_thresholded)
X_test_thresholded = np.array(X_test_thresholded)

# Define the background model
background_model = cv2.createBackgroundSubtractorMOG2()

# Apply the background model to the training and testing sets
X_train_foreground = []
X_test_foreground = []

for image in X_train_thresholded:
    foreground_mask = background_model.apply(image)
    foreground_image = cv2.bitwise_and(image, image, mask=foreground_mask)
    X_train_foreground.append(foreground_image)

for image in X_test_thresholded:
    foreground_mask = background_model.apply(image)
    foreground_image = cv2.bitwise_and(image, image, mask=foreground_mask)
    X_test_foreground.append(foreground_image)

# Convert the lists to NumPy arrays
X_train_foreground = np.array(X_train_foreground)
X_test_foreground = np.array(X_test_foreground)

# View the dataset
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train_foreground[i], cmap="gray")
    plt.xlabel(label_encoder.inverse_transform([y_train[i]]))
plt.show()

# Define the leaf segmentation model
leaf_segmentation_model = Sequential()
leaf_segmentation_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
leaf_segmentation_model.add(MaxPooling2D((2, 2)))
leaf_segmentation_model.add(Conv2D(64, (3, 3), activation='relu'))
leaf_segmentation_model.add(MaxPooling2D((2, 2)))
leaf_segmentation_model.add(Conv2D(128, (3, 3), activation='relu'))
leaf_segmentation_model.add(MaxPooling2D((2, 2)))
leaf_segmentation_model.add(Flatten())
leaf_segmentation_model.add(Dense(128, activation='relu'))
leaf_segmentation_model.add(Dense(1, activation='sigmoid'))

# Compile the model
leaf_segmentation_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
leaf_segmentation_model.fit(X_train_foreground, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = leaf_segmentation_model.evaluate(X_test_foreground, y_test)
print('Loss:', loss)
print('Accuracy:', accuracy)




# Extract features from the training and testing sets
X_train_features = base_model.predict(X_train_foreground)
X_test_features = base_model.predict(X_test_foreground)

# Flatten the features
X_train_features = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features = X_test_features.reshape(X_test_features.shape[0], -1)

# Normalize the features
X_train_features = X_train_features / X_train_features.max(axis=0)
X_test_features = X_test_features / X_test_features.max(axis=0)

# View the features
print(X_train_features[0])


X, y = [], []
for category in categories:
    path = os.path.join(data_directory, category)
    class_num = label_encoder.transform([category])[0]
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            resized_array = cv2.resize(img_array, (100, 100))
            X.append(resized_array)
            y.append(class_num)
        except Exception as e:
            pass


X = np.array(X) / 255.0  # Normalize pixel values
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(len(categories), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Calculate precision, recall, accuracy and f1 score
precision = precision_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1), average="macro")
recall = recall_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1), average="macro")
accuracy = accuracy_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1))
f1 = f1_score(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1), average="macro")

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Save the model
model.save('my_model.keras')