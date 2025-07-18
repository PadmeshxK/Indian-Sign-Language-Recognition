import os
import cv2
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
import string

from google import genai


client = genai.Client(api_key="AIzaSyBUOnAZwHmzhVYQMgbqK1lZRMLC976cCRA")

MODEL_NAME = "gemini-2.0-flash"  

from tensorflow import keras
model = keras.models.load_model("model.h5")

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands          = mp.solutions.hands

alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

debounce_frames = 7
last_label      = None
debounce_count  = 0
gesture_buffer  = []


def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [
        [min(int(lm.x * w), w-1), min(int(lm.y * h), h-1)]
        for lm in landmarks.landmark
    ]


def pre_process_landmark(pts):
    pts = copy.deepcopy(pts)
    base_x, base_y = pts[0]
    for i, (x, y) in enumerate(pts):
        pts[i][0] -= base_x
        pts[i][1] -= base_y
    flat = list(itertools.chain.from_iterable(pts))
    m = max(map(abs, flat))
    return [v / m for v in flat]


def gestures_to_sentence(seq: str) -> str:
    #Using nlp to convert detected alphabets into english and tamil
    system_prompt = (
        "You correct sign language letter alphabets into meaningful English. "
        "There might be characters incorrectly present between a word (e.g. 'hdlzo' -> 'hlo'). "
        "Prioritize real words and grammatical correctness. "
        "Return only the corrected sentence"
    )

    user_prompt = (
        f"Correct this potentially misrecognized sequence:\n"
        f"{' '.join(seq)}\n\n"
        f"Return ONLY the final sentence, then on a new line the Tamil translation which is in english(so vankkaam for hello translation)."
        f"Also return the final sentence in hindi too which is pronounced in english"
    )

    full_prompt = system_prompt + "\n\n" + user_prompt

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt
    )
    return resp.text.strip()


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                pts  = calc_landmark_list(canvas, lm)
                feat = pre_process_landmark(pts)
                df   = pd.DataFrame(feat).T

                preds = model.predict(df, verbose=0)
                lbl   = alphabet[np.argmax(preds)]

                if lbl == last_label:
                    debounce_count += 1
                else:
                    last_label     = lbl
                    debounce_count = 1

                if debounce_count == debounce_frames:
                    gesture_buffer.append(lbl)

                mp_drawing.draw_landmarks(
                    canvas, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                cv2.putText(canvas, lbl, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0,0,255), 2)

        cv2.putText(canvas, f"Buffer: {' '.join(gesture_buffer)}",
                    (10,70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,0), 2)
        cv2.putText(canvas, "Press 'S' to translate, 'ESC' to quit",
                    (10,110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        cv2.imshow('ISL → Gemini', canvas)

        key = cv2.waitKey(5) & 0xFF
        if key == 27:           # ESC to quit
            break
        elif key == ord('s') and gesture_buffer:
            sentence = gestures_to_sentence(' '.join(gesture_buffer))
            print("🡒", sentence)
            gesture_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()
