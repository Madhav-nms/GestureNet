import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Load model and scaler
with open('gesture_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

GESTURES = {
    0: "Open Palm",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace Sign",
    4: "Pointing"
}

COLORS = {
    0: (255, 165, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 255),
    4: (255, 255, 0)
}

def normalize_landmarks(hand_landmarks):
    wrist_x = hand_landmarks.landmark[0].x
    wrist_y = hand_landmarks.landmark[0].y
    wrist_z = hand_landmarks.landmark[0].z

    coords = []
    for lm in hand_landmarks.landmark:
        coords.append([lm.x - wrist_x,
                       lm.y - wrist_y,
                       lm.z - wrist_z])

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

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

# Metrics tracking
frame_count = 0
fps_start_time = time.time()
inference_times = []
fps = 0

print("Live demo running... Press Q to quit")
print("Benchmarking FPS and inference latency...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Calculate FPS every 30 frames
    if frame_count % 30 == 0:
        elapsed = time.time() - fps_start_time
        fps = 30 / elapsed
        fps_start_time = time.time()

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            row = normalize_landmarks(hand_landmarks)

            if row is not None:
                # Measure inference latency
                infer_start = time.perf_counter()
                row_scaled = scaler.transform([row])
                prediction = model.predict(row_scaled)[0]
                confidence = model.predict_proba(row_scaled)[0][prediction] * 100
                infer_end = time.perf_counter()

                # Track inference time in ms
                inference_ms = (infer_end - infer_start) * 1000
                inference_times.append(inference_ms)

                gesture_name = GESTURES[prediction]
                color = COLORS[prediction]

                # Display gesture and confidence
                cv2.putText(frame, f"{gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Display live metrics
                cv2.putText(frame, f"Inference: {inference_ms:.1f}ms", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Gesture Recognition - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Print final benchmark summary
cap.release()
cv2.destroyAllWindows()

if inference_times:
    print("\n=BENCHMARK SUMMARY =")
    print(f"Total frames processed : {frame_count}")
    print(f"Average FPS            : {fps:.1f}")
    print(f"Avg inference latency  : {np.mean(inference_times):.2f}ms")
    print(f"Min inference latency  : {np.min(inference_times):.2f}ms")
    print(f"Max inference latency  : {np.max(inference_times):.2f}ms")
    print(f"95th percentile latency: {np.percentile(inference_times, 95):.2f}ms")
    print("=====================================")