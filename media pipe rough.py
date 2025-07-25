import cv2
import mediapipe as mp

cap = cv2.VideoCapture(1)

mp.solutions.hands.Hands()
mp.solutions.drawing_utils

while True:
    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = mp.solutions.hands.process(image)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, handLms)

    cv2.imshow("Hand", image)

    if cv2.waitKey(10) == 113:
        break

cap.close()
