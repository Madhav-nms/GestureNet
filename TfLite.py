import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os
import time

print(f"TensorFlow version: {tf.__version__}")

# Load sklearn model and scaler
with open('gesture_model.pkl', 'rb') as f:
    sklearn_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load data
df = pd.read_csv('gesture_data_augmented.csv')
X = df.drop('label', axis=1).values
y = df['label'].values
X_scaled = scaler.transform(X).astype(np.float32)

# Rebuild in Keras
print("\nStep 1: Rebuilding model in Keras...")

inputs = tf.keras.Input(shape=(63,), name='input')
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Copy weights from sklearn
sklearn_weights = sklearn_model.coefs_
sklearn_biases  = sklearn_model.intercepts_
model.layers[1].set_weights([sklearn_weights[0], sklearn_biases[0]])
model.layers[2].set_weights([sklearn_weights[1], sklearn_biases[1]])
model.layers[3].set_weights([sklearn_weights[2], sklearn_biases[2]])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

loss, accuracy = model.evaluate(X_scaled, y, verbose=0)
print(f"Keras model accuracy: {accuracy*100:.2f}%")

# Save as SavedModel format first
saved_model_path = 'gesture_saved_model'
model.export(saved_model_path)
print(f"Model exported to {saved_model_path}")

# Float32 TFLite 
print("\nStep 2: Converting to Float32 TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
tflite_model = converter.convert()

with open('gesture_model_float32.tflite', 'wb') as f:
    f.write(tflite_model)

float32_size = os.path.getsize('gesture_model_float32.tflite') / 1024
print(f"Float32 model size: {float32_size:.2f} KB")

# INT8 Quantized TFLite 
print("\nStep 3: Converting to INT8 quantized TFLite...")

def representative_dataset():
    for i in range(len(X_scaled)):
        yield [X_scaled[i].reshape(1, 63)]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

with open('gesture_model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

int8_size = os.path.getsize('gesture_model_int8.tflite') / 1024
print(f"INT8 model size: {int8_size:.2f} KB")

# Step 4: Benchmark INT8 
print("\nStep 4: Benchmarking INT8 inference...")
interpreter = tf.lite.Interpreter(model_path='gesture_model_int8.tflite')
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_scale, input_zero_point = input_details[0]['quantization']

correct = 0
inference_times = []

for i in range(len(X_scaled)):
    sample = X_scaled[i]
    sample_int8 = (sample / input_scale + input_zero_point).astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'],
                           sample_int8.reshape(1, 63))

    start = time.perf_counter()
    interpreter.invoke()
    end = time.perf_counter()
    inference_times.append((end - start) * 1000)

    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output)
    if prediction == y[i]:
        correct += 1

int8_accuracy = correct / len(y) * 100

# Step 5: Summary 
print("\n========= CONVERSION SUMMARY =========")
print(f"Keras  accuracy          : {accuracy*100:.2f}%")
print(f"INT8   accuracy          : {int8_accuracy:.2f}%")
print(f"Accuracy drop            : {accuracy*100 - int8_accuracy:.2f}%")
print(f"Float32 model size       : {float32_size:.2f} KB")
print(f"INT8    model size       : {int8_size:.2f} KB")
print(f"Size reduction           : {((float32_size-int8_size)/float32_size)*100:.1f}%")
print(f"Avg inference latency    : {np.mean(inference_times):.3f} ms")
print(f"95th percentile latency  : {np.percentile(inference_times, 95):.3f} ms")
print("======================================")
print("\n Models saved:")
print("   gesture_model_float32.tflite")
print("   gesture_model_int8.tflite")