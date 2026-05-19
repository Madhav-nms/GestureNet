import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# Gesture labels for display
GESTURES = {
    0: "Open Palm",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace Sign",
    4: "Pointing"
}

# Load data
df = pd.read_csv('gesture_data_augmented.csv')

X = df.drop('label', axis=1).values  # 63 features
y = df['label'].values                # gesture class

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
print("\nTraining model...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=500,
    random_state=42,
    verbose=True
)
model.fit(X_train, y_train)

# Evaluate
print("\n--- Evaluation ---")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred,
      target_names=list(GESTURES.values())))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
ticks = list(GESTURES.values())
plt.xticks(range(5), ticks, rotation=45)
plt.yticks(range(5), ticks)
plt.ylabel('Actual')
plt.xlabel('Predicted')

# Add numbers inside boxes
for i in range(5):
    for j in range(5):
        plt.text(j, i, str(cm[i][j]),
                 ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as confusion_matrix.png")

# Save model and scaler
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(" Model saved as gesture_model.pkl")
print(" Scaler saved as scaler.pkl")  