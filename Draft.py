import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Detect which fingers are up (handles left/right hand)
def fingers_up(hand_landmarks, hand_label):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb (x-direction check is flipped for left vs right hand)
    if hand_label == "Right":
        if ((hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x) or 
            (hand_landmarks.landmark[tip_ids[0]].y < hand_landmarks.landmark[tip_ids[0] - 1].y)):
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other fingers (y-direction is the same for both hands)
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

brightBar = 400  # Position for brightness bar
brightPer = 0    # Brightness percentage
pTime = 0        # For FPS

# Use MediaPipe Hands
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = hand_handedness.classification[0].label  # "Left" or "Right"

                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect finger states
                fingers = fingers_up(hand_landmarks, hand_label)

                # Determine gesture
                if fingers == [0, 1, 0, 0, 0]:
                    gesture = "Index Finger Up"
                elif fingers == [0, 0, 0, 0, 0]:
                    gesture = "Fist"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture = "Palm"
                elif fingers == [1, 1, 0, 0, 0]:
                    gesture = "OK"
                elif fingers == [1, 1, 0, 0, 1]:
                    gesture = "Yo!"
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "Thumbs Up"
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture = "Two Fingers"
                else:
                    gesture = "Other"

                # Show gesture text
                x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                cv2.putText(frame, hand_label + ": " + gesture, (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # ---- Brightness control (using thumb + index pinch) ----
                a1, b1 = int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0])
                a2, b2 = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
                ca, cb = (a1 + a2) // 2, (b1 + b2) // 2

                cv2.circle(frame, (a1, b1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (a2, b2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (a1, b1), (a2, b2), (255, 0, 255), 2)
                cv2.circle(frame, (ca, cb), 10, (255, 0, 255), cv2.FILLED)

                length = math.hypot(a2 - a1, b2 - b1)
                brightBar = np.interp(length, [50, 300], [400, 150])
                brightPer = np.interp(length, [50, 300], [0, 100])

                sbc.set_brightness(int(brightPer))  # Apply brightness

                # Brightness Bar UI
                cv2.rectangle(frame, (550, 150), (585, 400), (0, 0, 0), 4)
                cv2.rectangle(frame, (550, int(brightBar)), (585, 400), (255, 255, 0), cv2.FILLED)
                cv2.putText(frame, "Brightness: " + str(int(brightPer)) + "%", (440, 450),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240, 32, 160), 2)

                current_brightness = sbc.get_brightness()
                cv2.putText(frame, "Light set: " + str(int(current_brightness[0])), (440, 120),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (240, 32, 160), 2)


        # Show the frame
        cv2.imshow("Gesture Detection with Brightness Control", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
