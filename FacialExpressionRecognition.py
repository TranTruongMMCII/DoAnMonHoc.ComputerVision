# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:28:16 2020

@author: Truong
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np

from tqdm import tqdm
import json


from deepface.commons import functions

import Expression


def recognition(img_path, actions = [], models = {}, enforce_detection = True, detector_backend = 'opencv'):
    if type(img_path) == list:
        img_paths = img_path.copy()
        bulkProcess = True
    else:
        img_paths = [img_path]
        bulkProcess = False


    if len(actions) == 0:
        actions = ['emotion']


    if 'emotion' in actions:
        if 'emotion' in models:
            print("already built emotion model is passed")
            emotion_model = models['emotion']
        else:
            emotion_model = Expression.loadModel()

    resp_objects = []

    disable_option = False if len(img_paths) > 1 else True

    global_pbar = tqdm(range(0, len(img_paths)), desc='Recognition progress', disable=disable_option)

    # for img_path in img_paths:
    for j in global_pbar:
        img_path = img_paths[j]

        resp_obj = "{"

        disable_option = False if len(actions) > 1 else True

        pbar = tqdm(range(0, len(actions)), desc='Finding actions', disable=disable_option)

        action_idx = 0

        # for action in actions:
        for index in pbar:
            action = actions[index]
            pbar.set_description("Action: %s" % (action))

            if action_idx > 0:
                resp_obj += ", "

            if action == 'emotion':
                emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                img = functions.preprocess_face(img=img_path, target_size=(48, 48), grayscale=True,
                                                enforce_detection=enforce_detection, detector_backend=detector_backend)

                emotion_predictions = emotion_model.predict(img)[0, :]

                sum_of_predictions = emotion_predictions.sum()

                emotion_obj = "\"emotion\": {"
                for i in range(0, len(emotion_labels)):
                    emotion_label = emotion_labels[i]
                    emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions

                    if i > 0: emotion_obj += ", "

                    emotion_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)

                emotion_obj += "}"

                emotion_obj += ", \"dominant_emotion\": \"%s\"" % (emotion_labels[np.argmax(emotion_predictions)])

                resp_obj += emotion_obj

            action_idx = action_idx + 1

        resp_obj += "}"

        resp_obj = json.loads(resp_obj)

        if bulkProcess == True:
            resp_objects.append(resp_obj)
        else:
            return resp_obj

    if bulkProcess == True:
        resp_obj = "{"

        for i in range(0, len(resp_objects)):
            resp_item = json.dumps(resp_objects[i])

            if i > 0:
                resp_obj += ", "

            resp_obj += "\"instance_" + str(i + 1) + "\": " + resp_item
        resp_obj += "}"
        resp_obj = json.loads(resp_obj)
        return resp_obj