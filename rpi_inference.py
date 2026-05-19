import cv2
import numpy as np
import pickle
import tflite_runtime.interpreter as tflite
import mediapipe as mp
import time
import subprocess
import threading
import queue

# Load TFLite model
interpreter = tflite.Interpreter(model_path='gesture_model_int8.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_scale, input_zero_point = input_details[0]['quantization']

# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

GESTURES = {
    0: "Open Palm",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace Sign",
    4: "Pointing"
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

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Frame queue for thread safe camera access
frame_queue = queue.Queue(maxsize=2)

def camera_thread():
    """Capture frames from rpicam-vid and put in queue"""
    cmd = [
        'rpicam-vid',
        '--codec', 'mjpeg',
        '--width', '640',
        '--height', '480',
        '--framerate', '30',
        '--timeout', '0',
        '--output', '-',
        '--nopreview'
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buffer = b''
    
    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break
        buffer += chunk
        
        # Find JPEG frames in buffer
        start = buffer.find(b'\xff\xd8')
        end = buffer.find(b'\xff\xd9')
        
        if start != -1 and end != -1 and end > start:
            jpg_data = buffer[start:end+2]
            buffer = buffer[end+2:]
            
            # Decode JPEG to numpy array
            frame = cv2.imdecode(
                np.frombuffer(jpg_data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if frame is not None:
                # Drop old frame if queue full
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put(frame)

# Start camera thread
print("Initializing camera...")
cam_thread = threading.Thread(target=camera_thread, daemon=True)
cam_thread.start()
time.sleep(3)  # Wait for camera to warm up
print("Camera ready!")

print("RPi Gesture Recognition Running...")
print("Press Ctrl+C to stop\n")

frame_count = 0
fps_start = time.time()
inference_times = []
fps = 0

try:
    while True:
        # Get frame from queue
        try:
            frame = frame_queue.get(timeout=2)
        except queue.Empty:
            print("Waiting for camera...")
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Calculate FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = 30 / elapsed
            fps_start = time.time()

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                row = normalize_landmarks(hand_landmarks)

                if row is not None:
                    row_scaled = scaler.transform([row]).astype(np.float32)

                    # Quantize input to INT8
                    row_int8 = (row_scaled / input_scale + input_zero_point).astype(np.int8)
                    interpreter.set_tensor(input_details[0]['index'], row_int8.reshape(1, 63))

                    # Run inference
                    infer_start = time.perf_counter()
                    interpreter.invoke()
                    infer_end = time.perf_counter()

                    inference_ms = (infer_end - infer_start) * 1000
                    inference_times.append(inference_ms)

                    output = interpreter.get_tensor(output_details[0]['index'])
                    prediction = np.argmax(output)
                    gesture_name = GESTURES[prediction]

                    print(f"Gesture: {gesture_name:12} | Inference: {inference_ms:.2f}ms | FPS: {fps:.1f}")

        else:
            print(f"No hand detected | FPS: {fps:.1f}")

except KeyboardInterrupt:
    pass

finally:
    if inference_times:
        print("\n========= RPi BENCHMARK =========")
        print(f"Total frames       : {frame_count}")
        print(f"Avg FPS            : {fps:.1f}")
        print(f"Avg inference      : {np.mean(inference_times):.2f}ms")
        print(f"95th percentile    : {np.percentile(inference_times, 95):.2f}ms")
        print(f"Min inference      : {np.min(inference_times):.2f}ms")
        print(f"Max inference      : {np.max(inference_times):.2f}ms")
        print("==================================")