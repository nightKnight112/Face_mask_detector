from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# preprocessing part

# initializing the initial learning rate, number of epochs to train for,
# and batch size
ini_lrn_rt = 0.0001
#epochs = 20  #ideal no of epochs for best accuracy
batch_size = 32

print("Enter the number epochs to be done", "integer input only", sep="\n", end="\n")
epochs = int(input())  #should be within 10-20 as less than 10 would cause a significant loss in accuracy, and greater than 20 does not cause a significant increase in accuracy

# dataset dir
dirs = r"D:\Sparks Foundation Internship Work\Project_2\Face_mask_detector\dataset_img_msk"
categories = ["with_mask", "without_mask"]

data, labels = [], []

for cat in categories:
    path = os.path.join(dirs, cat)
    print(path)
    for imgs in os.listdir(path):
        img_path = os.path.join(path, imgs)
        img = load_img(img_path, color_mode='rgb', target_size=(224, 224))
        img = img_to_array(img)
        img = preprocess_input(img)
        data.append(img)
        labels.append(cat)

# converting label list into bool type list....either 1 or 0
l_b = LabelBinarizer()
labels = l_b.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=54)

# to create many imgs frm the same img, we use ImageDataGenerator
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")


# modelling part
# pre-trained model "imagenet" used as baseModel

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)  # adding a dense layer of 128 perceptrons, "relu" as the use case is non linear
headModel = Dropout(0.5)(headModel)  # to avoid overfitting of the model
headModel = Dense(2, activation="softmax")(headModel)  # "softmax" activation as the classification is binary

model = Model(inputs=baseModel.input, outputs=headModel)

# looping over all layers in the base model and freezing them so they will not be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False


#compilation part

opt = Adam(lr=ini_lrn_rt, decay=ini_lrn_rt / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


#model fitting

# train the head of the network
print("training head of model", end="\n")
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size, validation_data=(testX, testY), validation_steps=len(testX) // batch_size, epochs=epochs)

# make predictions on the testing set
print("evaluating the network", end="\n")
predIdxs = model.predict(testX, batch_size=batch_size)
predIdxs = np.argmax(predIdxs, axis=1)


#classification report
print("Displaying Classification report: ", end="\n")
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=l_b.classes_))


print("saving mask detector model...", end="\n")
model.save("mask_detector_final.model", save_format="h5")


# plotting the training loss and accuracy
print("Plotting the training loss and accuracy and saving it", end="\n")

N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot_model_accuracy.png")

print("Code finished, model ready")