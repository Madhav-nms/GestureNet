import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

print("Starting hand tracker... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe needs RGB, OpenCV gives BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb_frame)

    # If hand detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks on screen
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Print all 21 landmark coordinates
            for idx, lm in enumerate(hand_landmarks.landmark):
                print(f"Landmark {idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

    # Show frame
    cv2.imshow("Hand Tracker", frame)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()