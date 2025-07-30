# ========== Import Required Libraries ==========
import cv2                             # OpenCV for webcam input and display
import mediapipe as mp                 # MediaPipe for hand tracking
import socketio                        # Socket.IO for sending data to server

# ========== Connect to Socket.IO Server ==========
sio = socketio.Client()                # Create a Socket.IO client
sio.connect('http://localhost:3000')   # Connect to the local Socket.IO server

# ========== Initialize MediaPipe Hands ==========
mp_hands = mp.solutions.hands                          # Access MediaPipe hands module
mp_drawing = mp.solutions.drawing_utils                # Utility for drawing hand landmarks

# ========== Function to Detect Which Fingers Are Up ==========
def fingers_up(hand_landmarks, hand_label):
    fingers = []                        # List to store finger states (1 for up, 0 for down)
    tip_ids = [4, 8, 12, 16, 20]        # Landmark indices for thumb, index, middle, ring, pinky tips

    # ----- Thumb -----
    # Thumb uses x-axis comparison; direction depends on left/right hand
    if hand_label == "Right":
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is down
    else:  # Left hand
        if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # ----- Other Fingers -----
    # Use y-axis to detect if finger tips are above middle joints
    for id in range(1, 5):
        if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)  # Finger is down

    return fingers

# ========== Start Webcam Video Capture ==========
cap = cv2.VideoCapture(0)                        # Start capturing from default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)           # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)          # Set frame height

# ========== Main Loop for Hand Detection ==========
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.75) as hands:
    while cap.isOpened():
        success, frame = cap.read()              # Read a frame from webcam
        if not success:
            break                                # If frame not read, exit the loop

        frame = cv2.flip(frame, 1)               # Flip the frame horizontally (mirror image)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for MediaPipe
        rgb_frame.flags.writeable = False        # Improve performance by disabling writing
        result = hands.process(rgb_frame)        # Process frame to detect hands
        rgb_frame.flags.writeable = True         # Re-enable writing

        # ========== If Hands Are Detected ==========
        if result.multi_hand_landmarks and result.multi_handedness:
            # Loop through each detected hand
            for i, (hand_landmarks, hand_handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
                hand_label = hand_handedness.classification[0].label  # Get hand label: "Left" or "Right"

                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect finger status (up/down)
                fingers = fingers_up(hand_landmarks, hand_label)

                # ========== Match Gesture Pattern ==========
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
                    gesture = "Other"  # Unknown/Unhandled gesture

                # ========== Send Gesture Data to Server ==========
                sio.emit('gesture', {'hand': hand_label, 'gesture': gesture})

        # ========== Display Output Frame ==========
        cv2.imshow("Two-Hand Gesture Detection", frame)  # Show the annotated video

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ========== Release Resources ==========
cap.release()                         # Stop video capture
cv2.destroyAllWindows()              # Close all OpenCV windows