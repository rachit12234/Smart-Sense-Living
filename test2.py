import cv2 as cv
import mediapipe as mp
import pyautogui as pag
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)
mp_lines = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pag.size()

# Smoothening variables
prev_x, prev_y = 0, 0
smoothening = 7

# Click control
last_click_time = 0
click_cooldown = 1  # seconds

def fingers_up(hand_landmarks, hand_label):
    fingers = []
    thumb_tip = 4
    thumb_tip_x = hand_landmarks.landmark[thumb_tip].x
    finger_tips = [8, 12, 16, 20]

    if hand_label == "Right":
        fingers.append(1 if thumb_tip_x < hand_landmarks.landmark[thumb_tip - 2].x else 0)
    else:
        fingers.append(1 if thumb_tip_x > hand_landmarks.landmark[thumb_tip - 2].x else 0)

    for tip in finger_tips:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    return fingers

def mouse_control(current_gesture, hand_landmarks):
    global prev_x, prev_y, last_click_time

    # Cursor move when only index finger is up
    if current_gesture == [0, 1, 0, 0, 0]:
        index_tip = hand_landmarks.landmark[8]
        x = int(index_tip.x * screen_width)
        y = int(index_tip.y * screen_height)

        smooth_x = prev_x + (x - prev_x) // smoothening
        smooth_y = prev_y + (y - prev_y) // smoothening

        pag.moveTo(smooth_x, smooth_y)
        prev_x, prev_y = smooth_x, smooth_y

    # Left click when index finger is down (tip below joint)
    if current_gesture[1] == 0:
        current_time = time.time()
        if current_time - last_click_time > click_cooldown:
            pag.click()
            last_click_time = current_time

# Webcam setup
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1300)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 700)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    result = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if result.multi_hand_landmarks and result.multi_handedness:
        for i, (hand_landmarks, hand_handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            hand_label = hand_handedness.classification[0].label
            mp_lines.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_states = fingers_up(hand_landmarks, hand_label)

            cv.putText(frame,
                       f'{hand_label} Hand: {sum(finger_states)} fingers up',
                       (10, 50 + i * 30),
                       cv.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2)

            mouse_control(finger_states, hand_landmarks)

    cv.imshow("Webcam", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()