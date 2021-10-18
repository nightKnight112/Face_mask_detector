from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import sys

#loading models
prototxtPath = r"face_detector_model\deploy.prototxt"
weightsPath = r"face_detector_model\res10_300x300_ssd_iter_140000.caffemodel"
face_model = cv2.dnn.readNet(prototxtPath, weightsPath)

#for best accuracy, train model with 20 epochs or more
mask_model = load_model("mask_detector_final.model")


def face_mask_detector(frm, faceNet, maskNet):
    h_img, w_img = frm.shape[:2]
    blob = cv2.dnn.blobFromImage(frm, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # passing the blob through the network and obtaining the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        # extracting the confidence associated with the detection
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:    # filtering out weak detections

            box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])   # bounding box computation for the object
            startX, startY, endX, endY = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w_img - 1, endX), min(h_img - 1, endY))

            # extracting the face ROI

            face = frm[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # adding the face and bounding boxes to their respective lists
            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        # making batch predictions for faster inference
        faces = np.array(faces, dtype="float32")
        predictions = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return locations, predictions

capture = cv2.VideoCapture(0)
capture.set(4, 400)

print("Starting VideoCapture, press 'E' to stop", end='\n')

while True:
    ret, frm = capture.read()
    if ret is False:
        print("Video not provided, exitting program")
        break
    locations, predictions = face_mask_detector(frm, face_model, mask_model)

    for (box, pred) in zip(locations, predictions):

        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask Worn" if mask > withoutMask else "No Mask Worn"
        color = (255, 0, 0) if label == "Mask" else (0, 0, 255)

        # including the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        cv2.putText(frm, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frm, (startX, startY), (endX, endY), color, 3)

    cv2.imshow("final_display", frm)

    #exit sequence
    if cv2.waitKey(1) & 0xff == ord('e'):
        print("...Exitting program...", end="\n")
        break

capture.release()
cv2.destroyAllWindows()
sys.exit()
