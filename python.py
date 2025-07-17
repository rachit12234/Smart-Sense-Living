import cv2
import mediapipe as mp
import socketio

sio = socketio.Client()
sio.connect('http://localhost:3000')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Detect which fingers are up (handles left/right hand)
def fingers_up(hand_landmarks, hand_label):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    # Thumb (x-direction check is flipped for left vs right hand)
    if hand_label == "Right":
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
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
            for i, (hand_landmarks, hand_handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
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
                elif fingers == [0, 0, 1, 0, 0]:
                    gesture = "Middle Finger"
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

                sio.emit('gesture', {'hand': hand_label, 'gesture': gesture})

                # Show gesture on frame
                x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                cv2.putText(frame, f'{hand_label}: {gesture}', (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Two-Hand Gesture Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()