import pandas as pd
import numpy as np

INPUT_FILE = 'gesture_data.csv'
OUTPUT_FILE = 'gesture_data_augmented.csv'

# Load original data
df = pd.read_csv('INPUT_FILE')
print(f"Original samples: {len(df)}")

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Mirror augmentation
# For each landmark, x coordinate is at positions 0, 3, 6, 9... (every 3rd starting at 0)
# Mirroring = flipping x relative to wrist
# Since we normalized relative to wrist already, we just negate all x values

mirrored_rows = []
for i in range(len(X)):
    row = X[i].copy()
    # Negate every x coordinate (positions 0, 3, 6, ... 60)
    for j in range(0, 63, 3):
        row[j] = -row[j]
    mirrored_rows.append(row)

mirrored_X = np.array(mirrored_rows)

# Build mirrored dataframe
columns = df.columns.tolist()
mirrored_df = pd.DataFrame(mirrored_X, columns=columns[:-1])
mirrored_df['label'] = y

# Combine original + mirrored
augmented_df = pd.concat([df, mirrored_df], ignore_index=True)
augmented_df.to_csv(OUTPUT_FILE, index=False)

print(f"Augmented samples: {len(augmented_df)}")
print("Samples per gesture:")
print(augmented_df['label'].value_counts().sort_index())
print(f"\n Augmented data saved to {OUTPUT_FILE}")
