import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Gestures we're collecting
GESTURES = {
    0: "Open Palm",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace Sign",
    4: "Pointing"
}

def normalize_landmarks(hand_landmarks):
    # Get wrist position (landmark 0) as origin
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z

    # Translate all landmarks relative to wrist
    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x - wrist_x,
                       lm.y - wrist_y,
                       lm.z - wrist_z])

    # Scale by the distance between wrist and middle finger base (landmark 9)
    # This normalizes for hand size and distance from camera
    scale = np.sqrt(
        (coords[9][0])**2 +
        (coords[9][1])**2 +
        (coords[9][2])**2
    )

    if scale == 0:
        return None

    normalized = []
    for c in coords:
        normalized += [c[0]/scale, c[1]/scale, c[2]/scale]

    return normalized

SAMPLES_PER_GESTURE = 200
CSV_FILE = "gesture_data.csv"

# Create CSV with headers 
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks x 3 coordinates = 63 features + label
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

cap = cv2.VideoCapture(0)

for gesture_id, gesture_name in GESTURES.items():
    count = 0
    collecting = False

    print(f"\n--- Get ready for: {gesture_name} ---")
    print(f"Press SPACE to start collecting 200 samples")

    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Display status on screen
        status = f"Gesture: {gesture_name} | Samples: {count}/{SAMPLES_PER_GESTURE}"
        color = (0, 255, 0) if collecting else (0, 0, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if not collecting:
            cv2.putText(frame, "Press SPACE to start", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if collecting:
                    # normalized
                    row = normalize_landmarks(hand_landmarks)

                    if row is None:
                        continue
                    
                    row.append(gesture_id)

                    # Save to CSV
                    with open(CSV_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    count += 1

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            collecting = True
            print(f"Collecting {gesture_name}...")
        if key == ord('q'):
            break

    print(f" Done collecting {gesture_name} — {count} samples saved")

cap.release()
cv2.destroyAllWindows()
print("\n All gestures collected ! Check gesture_data.csv")
