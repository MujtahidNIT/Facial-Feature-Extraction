#!/usr/bin/env python3

import pandas as pd
import scipy.stats as stats

# Load your data from the CSV file
csv_file_path = "/home/mujtahid/MediapipeFacialDataset/facial_landmarks.csv"
df = pd.read_csv(csv_file_path)

# Compare X across Eyes Open vs. not open
eyes_x = df[df["Eyes_Open"]]["X"]
not_eyes_x = df[~df["Eyes_Open"]]["X"]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(eyes_x, not_eyes_x)
print(f"X: Eyes_Open vs. Not Eyes_Open - p-value: {p_value:.4f}")

# Compare Y across smiling vs. not smiling
eyes_y = df[df["Eyes_Open"]]["Y"]
not_eyes_y = df[~df["Eyes_Open"]]["Y"]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(eyes_y, not_eyes_y)
print(f"Y: Eyes_Open vs. Not Eyes_Open - p-value: {p_value:.4f}")

# Compare Z across smiling vs. not smiling
eyes_z = df[df["Eyes_Open"]]["Z"]
not_eyes_z = df[~df["Eyes_Open"]]["Z"]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(eyes_z, not_eyes_z)
print(f"Z: Eyes_Open vs. Not Eyes_Open - p-value: {p_value:.4f}")


# Compare X across mouth open vs. not open
mouth_open_x = df[df["Mouth_Open"]]["X"]
not_m_open_x = df[~df["Mouth_Open"]]["X"]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(mouth_open_x, not_m_open_x)
print(f"X: Mouth_Open vs. Not Mouth_Open - p-value: {p_value:.4f}")

# Compare Y across mouth open vs. not open
mouth_open_y = df[df["Mouth_Open"]]["Y"]
not_m_open_y = df[~df["Mouth_Open"]]["Y"]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(mouth_open_y, not_m_open_y)
print(f"Y: Mouth_Open vs. Not Mouth_Open - p-value: {p_value:.4f}")

# Compare Z across mouth open vs. not open
mouth_open_z = df[df["Mouth_Open"]]["Z"]
not_m_open_z = df[~df["Mouth_Open"]]["Z"]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(mouth_open_z, not_m_open_z)
print(f"Z: Mouth_Open vs. Not Mouth_Open - p-value: {p_value:.4f}")
