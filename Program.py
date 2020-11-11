# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:01:06 2020

@author: Truong
"""
# !pip install keras

import os
from os import listdir
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout
import gdown
import zipfile
from keras.preprocessing import image


# from FacialExpressionRecognition import *
import FacialExpressionRecognition

opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]
path = folders[0]
for folder in folders[1:]:
    path = path + "/" + folder

detector_path = path + "/data/haarcascade_frontalface_default.xml"

if os.path.isfile(detector_path) != True:
    raise ValueError("Confirm that opencv is installed on your environment! Expected path ", detector_path,
                     " violated.")
else:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detectFace(img_path, target_size=(152, 152)):
    img = cv2.imread(img_path)

    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    if len(faces) > 0:
        x, y, w, h = faces[0]

        margin = 0
        x_margin = w * margin / 100
        y_margin = h * margin / 100

        if y - y_margin > 0 and y + h + y_margin < img.shape[1] and x - x_margin > 0 and x + w + x_margin < img.shape[
            0]:
            detected_face = img[int(y - y_margin):int(y + h + y_margin), int(x - x_margin):int(x + w + x_margin)]
        else:
            detected_face = img[int(y):int(y + h), int(x):int(x + w)]

        detected_face = cv2.resize(detected_face, target_size)

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)


        img_pixels /= 255

        return img_pixels
    else:
        raise ValueError("Face could not be detected in ", img_path,
                         ". Please confirm that the picture is a face photo.")
        
def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def Main():
    # try:
        
    # except:
    #     print("Something went wrong")
    target_size = (152, 152)  
    
    base_model = Sequential()
    base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
    base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
    base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5'))
    base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
    base_model.add(Flatten(name='F0'))
    base_model.add(Dense(4096, activation='relu', name='F7'))
    base_model.add(Dropout(rate=0.5, name='D0'))
    base_model.add(Dense(8631, activation='softmax', name='F8'))
    
    if os.path.isfile('VGGFace2_DeepFace_weights_val-0.9034.h5') != True:
        print("VGGFace2_DeepFace_weights_val-0.9034.h5 will be downloaded...")
		

        url = 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
        output = 'VGGFace2_DeepFace_weights_val-0.9034.zip'
        gdown.download(url, output, quiet=False)
		

        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall()
    
    base_model.load_weights(r"VGGFace2_DeepFace_weights_val-0.9034.h5")
    
    # Drop F8 and D0 layers. F7 is the representation layer.
    model = Model(inputs=base_model.layers[0].input, outputs=base_model.layers[-3].output)
    
    
    
    user_pictures = "Image/"
    
    users = dict()
    
    for file in listdir(user_pictures):
        user, extension = file.split(".")
        img_path = 'Image/%s.png' % (user)
        img = detectFace(img_path)
    
        representation = model.predict(img)[0]
    
        users[user] = representation
    
    print("user representations retrieved successfully")
    
    # ------------------------
    
    cap = cv2.VideoCapture(0)  # webcam
    is_found = False

    
    while (True):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            if w > 130:  # discard small detected faces
                detect_face = img[int(y):int(y+h), int(x):int(x+w), 0:3]
                string = FacialExpressionRecognition.recognition(detect_face,enforce_detection=False)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # draw rectangle to main image
                cv2.putText(img, string["dominant_emotion"], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                detected_face = cv2.resize(detected_face, target_size)  # resize to 152x152
    
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255
    
                captured_representation = model.predict(img_pixels)[0]
    
                distances = []
    
                for i in users:
                    user_name = i
                    source_representation = users[i]
    
                    distance = findEuclideanDistance(l2_normalize(captured_representation),
                                                     l2_normalize(source_representation))
                    distances.append(distance)
    
                is_found = False
                index = 0
                for i in users:
                    user_name = i
                    if index == np.argmin(distances):
                            if distances[index] <= 0.70:
                                print("detected: ", user_name, "(", distances[index], ")")
                                user_name = user_name.replace("_", "")
                                similarity = distances[index]
        
                                is_found = True
    
                                break
    
                    index = index + 1
    
            if is_found:
                display_img = cv2.imread("Image/%s.png" % user_name)
                pivot_img_size = 112
                display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))
        
                try:
                    resolution_x = img.shape[1];
                    resolution_y = img.shape[0]
        
                    label = user_name + " (" + "{0:.2f}".format(similarity) + ")"
        
                    if y - pivot_img_size > 0 and x + w + pivot_img_size < resolution_x:
                        # top right
                        img[y - pivot_img_size:y, x + w:x + w + pivot_img_size] = display_img
                        cv2.putText(img, label, (x + w, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
                        # connect face and text
                        cv2.line(img, (x + int(w / 2), y), (x + 3 * int(w / 4), y - int(pivot_img_size / 2)), (0, 255, 0), 1)
                        cv2.line(img, (x + 3 * int(w / 4), y - int(pivot_img_size / 2)), (x + w, y - int(pivot_img_size / 2)),
                                 (0, 255, 0), 1)
                    elif y + h + pivot_img_size < resolution_y and x - pivot_img_size > 0:
                        # bottom left
                        img[y + h:y + h + pivot_img_size, x - pivot_img_size:x] = display_img
                        cv2.putText(img, label, (x - pivot_img_size, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                    1)
        
                        # connect face and text
                        cv2.line(img, (x + int(w / 2), y + h), (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                 (0, 255, 0), 1)
                        cv2.line(img, (x + int(w / 2) - int(w / 4), y + h + int(pivot_img_size / 2)),
                                 (x, y + h + int(pivot_img_size / 2)), (0, 255, 0), 1)
        
                    elif y - pivot_img_size > 0 and x - pivot_img_size > 0:
                        # top left
                        img[y - pivot_img_size:y, x - pivot_img_size:x] = display_img
                        cv2.putText(img, label, (x - pivot_img_size, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
                        # connect face and text
                        cv2.line(img, (x + int(w / 2), y), (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                 (0, 255, 0), 1)
                        cv2.line(img, (x + int(w / 2) - int(w / 4), y - int(pivot_img_size / 2)),
                                 (x, y - int(pivot_img_size / 2)), (0, 255, 0), 1)
        
                    elif x + w + pivot_img_size < resolution_x and y + h + pivot_img_size < resolution_y:
                        # bottom righ
                        img[y + h:y + h + pivot_img_size, x + w:x + w + pivot_img_size] = display_img
                        cv2.putText(img, label, (x + w, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
                        # connect face and text
                        cv2.line(img, (x + int(w / 2), y + h), (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                 (0, 255, 0), 1)
                        cv2.line(img, (x + int(w / 2) + int(w / 4), y + h + int(pivot_img_size / 2)),
                                 (x + w, y + h + int(pivot_img_size / 2)), (0, 255, 0), 1)
        
                except Exception as e:
                    print("exception occured: ", str(e))
            else:
                cv2.putText(img, "Unknown", (x + w, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
            cv2.imshow('img', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
    
    # kill open cv things		
    cap.release()
    cv2.destroyAllWindows()
    
# try:
#     Main()
# except:
#     print("something went wrong")
