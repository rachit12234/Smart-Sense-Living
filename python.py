import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def fingers_up(hand_landmarks):
    fingers = []

    tip_ids = [4, 8, 12, 16, 20]

    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)  
    else:
        fingers.append(0)

    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        result = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = fingers_up(hand_landmarks)

                if fingers == [0, 1, 0, 0, 0]:
                    gesture = "Index Finger Up"
                elif fingers == [0, 0, 0, 0, 0]:
                    gesture = "Fist"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture = "Palm"
                elif fingers == [0, 0, 1, 0, 0]:
                    gesture = "Fuck You too"
                elif fingers == [1, 1, 0, 0, 0]:
                    gesture = "Ok"
                elif fingers == [1, 1, 0, 0, 1]:
                    gesture = "Yo!"
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "Thumbs Up"
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture = "2 fingers"
                else:
                    gesture = "Other"

                cv2.putText(frame, f'Gesture: {gesture}', (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        cv2.imshow("Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
