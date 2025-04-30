# phase1_model_training.ipynb
import os
import numpy as np
import matplotlib.pyplot as plt
# Set project directories
BASE_DIR = os.path.abspath("..")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
# Quick check
print("Data Folder Exists:", os.path.exists(DATA_DIR))

import cv2
from glob import glob
from IPython.display import display
from PIL import Image
# Constants
IMAGE_SIZE = (256, 256)  # You can upscale later
IMAGE_FOLDER = DATA_DIR
# Step 1: Load Images in Time Series Order
def load_images(folder_path):
    image_paths = sorted(glob(os.path.join(folder_path, "*.png")))
    images = []
    years = []
    
    for path in image_paths:
        try:
            year = int(os.path.basename(path).split('_')[-1].split('.')[0])
            img = cv2.imread(path)
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
            years.append(year)
        except:
            continue
    return np.array(images), years
images, years = load_images(DATA_DIR)
print(f"Total Years Loaded: {len(images)}")

def plot_sample_images(images, years, step=5):
    plt.figure(figsize=(15, 5))
    for i in range(0, len(images), step):
        plt.subplot(1, len(images[::step]), i//step + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(str(years[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
plot_sample_images(images, years, step=6)  # Show one every 6 years


SEQUENCE_LENGTH = 5  # You can experiment with this later
def create_time_series_data(images, sequence_length):
    X, y = [], []
    for i in range(len(images) - sequence_length):
        X.append(images[i:i+sequence_length])
        y.append(images[i+sequence_length])
    return np.array(X), np.array(y)
X, y = create_time_series_data(images, SEQUENCE_LENGTH)
print(f"Input shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")


# Assuming you have a 'years' list that corresponds to the images
# For example:
years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
# Create time-series years data
sequence_years = [years[i:i+SEQUENCE_LENGTH] for i in range(len(years) - SEQUENCE_LENGTH)]
sequence_years = sequence_years[:len(X)]  # Align this with X after augmentations
# Now 'sequence_years' should be aligned with 'X' and ready to use
print(f"Years shape (sequence_years): {np.array(sequence_years).shape}")


def plot_sequence_with_target(X, y, index=0):
    plt.figure(figsize=(15, 3))
    
    for i in range(SEQUENCE_LENGTH):
        plt.subplot(1, SEQUENCE_LENGTH + 1, i + 1)
        plt.imshow(cv2.cvtColor(X[index][i], cv2.COLOR_BGR2RGB))
        plt.title(f"Year {years[index + i]}")
        plt.axis('off')
    
    plt.subplot(1, SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH + 1)
    plt.imshow(cv2.cvtColor(y[index], cv2.COLOR_BGR2RGB))
    plt.title(f"Target {years[index + SEQUENCE_LENGTH]}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
plot_sequence_with_target(X, y)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Dense, 
                                     TimeDistributed, ConvLSTM2D, Reshape, BatchNormalization, Dropout,
                                     Flatten, UpSampling2D)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
IMAGE_SIZE = (256, 256)
CHANNELS = 3
SEQUENCE_LENGTH = 5
# --- Loss and Metrics ---
def combined_ssim_mse_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim = tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=2.0))  # for [-1, 1] range
    return 0.1 * mse + (1 - ssim)  # SSIM is given more weight
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=2.0))
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=2.0))
# --- Model Definition ---
def build_model(sequence_length, height, width, channels):
    inputs = Input(shape=(sequence_length, height, width, channels))
    # Encoder
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), name='conv_1')(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name='pool_1')(x)
    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'), name='conv_2')(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)), name='pool_2')(x)
    # ConvLSTM Stack
    x = ConvLSTM2D(64, (3, 3), activation='tanh', padding='same', return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = ConvLSTM2D(32, (3, 3), activation='tanh', padding='same', return_sequences=False)(x)
    x = Dropout(0.3)(x)
    # Decoder with UpSampling
    x = UpSampling2D(size=(2, 2))(x)  # Upsample to 128x128
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2, 2))(x)  # Upsample to 256x256
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    # Output Layer
    output = Conv2D(3, (3, 3), activation='tanh', padding='same')(x)
    # Model Compile
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        loss=combined_ssim_mse_loss,
        optimizer=Adam(learning_rate=0.001),
        metrics=['mae', ssim_metric, psnr_metric]
    )
    
    return model
    
model = build_model(SEQUENCE_LENGTH, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
model.summary()

# --- Data Normalization ---
X = (X - 127.5) / 127.5  # Normalize inputs to [-1, 1]
y = (y - 127.5) / 127.5  # Already normalized to [-1, 1]


def augment_data(X, y):
    X_aug = []
    y_aug = []
    for i in range(X.shape[0]):
        seq = X[i]
        target = y[i]
        # Random horizontal flip
        if np.random.rand() > 0.5:
            seq = np.flip(seq, axis=2)
            target = np.flip(target, axis=1)
        # Random vertical flip
        if np.random.rand() > 0.5:
            seq = np.flip(seq, axis=1)
            target = np.flip(target, axis=0)
        # Random rotation (90° steps only to avoid distortion)
        k = np.random.choice([0, 1, 2, 3])
        seq = np.rot90(seq, k=k, axes=(1, 2))
        target = np.rot90(target, k=k, axes=(0, 1))
        # Brightness jitter (small, to simulate seasonal/lighting change)
        brightness_factor = np.random.uniform(0.9, 1.1)
        seq = np.clip(seq * brightness_factor, -1.0, 1.0)
        target = np.clip(target * brightness_factor, -1.0, 1.0)
        X_aug.append(seq)
        y_aug.append(target)
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    # Concatenate original and augmented data
    X_final = np.concatenate([X, X_aug])
    y_final = np.concatenate([y, y_aug])
    return X_final, y_final
AUGMENT = True
if AUGMENT:
    X_augmented, y_augmented = augment_data(X, y)


# Get how many times the original data was expanded by augmentation
repeat_factor = X_augmented.shape[0] // X.shape[0]
# Now tile/replicate based on the repeat_factor
sequence_years_augmented = sequence_years * repeat_factor
print("Original X shape:", X.shape)
print("Augmented X shape:", X_augmented.shape)
print("Repeat factor:", repeat_factor)
print("Augmented Years shape:", np.shape(sequence_years_augmented))  # should match X_augmented.shape[0]


# --- Data Split ---
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented, test_size=0.2, random_state=42)
# Extract year sequences before splitting
sequence_years = [years[i:i+SEQUENCE_LENGTH] for i in range(len(years)-SEQUENCE_LENGTH)]
# Split both data and year sequences together (ensure aligned)
X_train, X_test, y_train, y_test, X_train_years, X_test_years = train_test_split(
    X, y, sequence_years, test_size=0.2, random_state=42
)


from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback, ModelCheckpoint
# --- Callbacks ---
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.h5"),
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)
def print_predictions(epoch, logs):
    sample_preds = model.predict(X_test[:1])
    print(f"\nSample prediction summary at epoch {epoch + 1}: mean={np.mean(sample_preds):.4f}, std={np.std(sample_preds):.4f}")
monitor_callback = LambdaCallback(on_epoch_end=print_predictions)
# --- Training ---
history = model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=4,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint_cb, monitor_callback]
)


model.save(os.path.join(MODEL_DIR, "cnn_lstm_deforestation_model.h5"))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.title("Training vs Validation Loss")
plt.show()


# Pick a sample from training set to warm up model
sample_sequence = X_train[0:1]  # Shape: (1, 5, 256, 256, 3)
_ = model.predict(sample_sequence)

def visualize_feature_maps(model, sample_sequence, layer_names, num_features=3, time_step=0):
    from tensorflow.keras.models import Model
    import matplotlib.pyplot as plt
    # Define a model that outputs intermediate layers
    intermediate_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(name).output for name in layer_names]
    )
    # Get outputs from the specified layers
    outputs = intermediate_model.predict(sample_sequence)
    for output, name in zip(outputs, layer_names):
        print(f"\nVisualizing Layer: {name}, Output shape: {output.shape}")
        
        fig, axs = plt.subplots(1, num_features, figsize=(15, 5))
        fig.suptitle(f"Feature Maps from Layer: {name}", fontsize=14)
        for i in range(num_features):
            # Decide if it's time-distributed or not
            if len(output.shape) == 5:  # (batch, time, h, w, c)
                feature_map = output[0, time_step, :, :, i]
            else:  # (batch, h, w, c)
                feature_map = output[0, :, :, i]
            axs[i].imshow(feature_map, cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f'Feature {i}')
        plt.tight_layout()
        plt.show()


sample_sequence = X_train[0:1]  # shape: (1, 5, 256, 256, 3)
layer_names = ['pool_1', 'pool_2']  # or ['conv_1', 'conv_2'] if you want
visualize_feature_maps(model, sample_sequence, layer_names, num_features=3, time_step=0)


# Predict future images
y_pred = model.predict(X_test)
# Reshape predictions back to image format
y_pred_images = y_pred.reshape(-1, 256, 256, 3)
y_true_images = y_test.reshape(-1, 256, 256, 3)


import os
import matplotlib.pyplot as plt
# Make sure output folder exists
BASE_OUTPUT = os.path.join("..", "outputs")
predicted_image_folder = os.path.join(BASE_OUTPUT, "predicted_images")
os.makedirs(predicted_image_folder, exist_ok=True)
def denormalize(image):
    return ((image + 1.0) / 2.0).clip(0, 1)  # Scale to [0, 1]
def show_prediction_vs_actual(y_true, y_pred, index=0, input_years=None, save=True):
    plt.figure(figsize=(10, 5))
    if input_years:
        predicted_year = input_years[index][-1] + 1
        title_pred = f"Predicted (Year {predicted_year})"
        title_true = f"Actual (Year {predicted_year})"
        filename = f"prediction_vs_actual_{predicted_year}.png"
    else:
        title_pred = "Predicted"
        title_true = "Actual"
        filename = f"prediction_vs_actual_{index}.png"
    plt.subplot(1, 2, 1)
    plt.imshow(denormalize(y_true[index]))
    plt.title(title_true)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(denormalize(y_pred[index]))
    plt.title(title_pred)
    plt.axis('off')
    plt.tight_layout()
    
    # Save the comparison image
    if save:
        save_path = os.path.join(predicted_image_folder, filename)
        plt.savefig(save_path)
        print(f"✅ Saved comparison plot: {save_path}")
    
    plt.show()
# Optional: Extract corresponding input years before splitting
sequence_years = [years[i:i+SEQUENCE_LENGTH] for i in range(len(years)-SEQUENCE_LENGTH)]
_, X_test_years = train_test_split(sequence_years, test_size=0.2, random_state=42)
# Show prediction
show_prediction_vs_actual(y_true_images, y_pred_images, index=0, input_years=X_test_years)

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def evaluate_predictions(y_true_images, y_pred_images):
    total_ssim = 0
    total_psnr = 0
    total_samples = len(y_true_images)
    for i in range(total_samples):
        true_img = y_true_images[i]
        pred_img = y_pred_images[i]
        # Pass data_range=1.0 since images are normalized
        ssim_score = ssim(true_img, pred_img, channel_axis=2, data_range=1.0)
        psnr_score = psnr(true_img, pred_img, data_range=1.0)
        total_ssim += ssim_score
        total_psnr += psnr_score
    avg_ssim = total_ssim / total_samples
    avg_psnr = total_psnr / total_samples
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
y_pred_images = np.clip(y_pred_images, -1.0, 1.0)
evaluate_predictions(y_true_images, y_pred_images)


def predict_future_ndvi(model, initial_sequence, num_future_years=3):
    """
    Predicts future NDVI maps using a rolling window of CNN-LSTM.
    :param model: trained model
    :param initial_sequence: numpy array of shape (5, 256, 256, 3)
    :param num_future_years: number of years to predict
    :return: list of predicted NDVI images
    """
    predictions = []
    current_sequence = initial_sequence.copy()
    for _ in range(num_future_years):
        input_seq = current_sequence[np.newaxis, ...]  # shape: (1, 5, 256, 256, 3)
        pred_image = model.predict(input_seq)[0]  # shape: (256, 256, 3)
        pred_image = np.clip(pred_image, -1.0, 1.0)  # Ensure in valid range
        predictions.append(pred_image)
        # Update sequence
        current_sequence = np.concatenate((current_sequence[1:], pred_image[np.newaxis, ...]), axis=0)
    return predictions
# Use last sequence from training/whole set (2020–2024 assumed)
last_sequence = X[-1]  # shape: (5, 256, 256, 3)
future_predictions = predict_future_ndvi(model, last_sequence, num_future_years=3)
# Future years
future_years = [2025, 2026, 2027]
# Plotting
plt.figure(figsize=(15, 5))
for i, img in enumerate(future_predictions):
    plt.subplot(1, len(future_predictions), i+1)
    plt.imshow(denormalize(img))  # if normalized to [-1, 1]
    plt.title(f"Predicted {future_years[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# Define where to save predicted NDVI .npy files
predicted_image_folder = os.path.join(BASE_OUTPUT, "predicted_images")
os.makedirs(predicted_image_folder, exist_ok=True)
# Save each predicted image
for i, pred_image in enumerate(future_predictions):
    year = future_years[i]
    # Save .npy for processing
    npy_path = os.path.join(predicted_image_folder, f"ndvi_{year}.npy")
    np.save(npy_path, pred_image)
    # Save .png for visualization
    png_path = os.path.join(predicted_image_folder, f"ndvi_{year}.png")
    plt.imsave(png_path, denormalize(pred_image), cmap='viridis')
print(f"✅ Saved predicted .npy and .png files in '{predicted_image_folder}'")
ndvi_2024 = last_sequence[-1]  # Extract NDVI image for 2024
# Save path
output_folder = os.path.join("..", "outputs", "predicted_images")
os.makedirs(output_folder, exist_ok=True)
# Save NDVI for 2024
np.save(os.path.join(output_folder, "ndvi_2024.npy"), ndvi_2024)
print("✅ NDVI file for 2024 saved successfully!")

import os
import numpy as np
import matplotlib.pyplot as plt
def compute_change_map(image1, image2, threshold=0.1):
    """
    Compute binary change map between two NDVI images.
    Highlights pixels where the mean channel-wise difference exceeds threshold.
    """
    diff = np.abs(image2 - image1)
    mask = np.mean(diff, axis=2) > threshold
    return mask
# Define project-level output directory (one level up from notebooks)
OUTPUT_DIR = os.path.join("..", "outputs", "predicted_changemaps")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Get the last known NDVI (2024)
previous_ndvi = X[-1][-1]
previous_ndvi = (previous_ndvi - previous_ndvi.min()) / (previous_ndvi.max() - previous_ndvi.min() + 1e-8)
future_years = [2025, 2026, 2027]
for i, year in enumerate(future_years):
    predicted_ndvi = future_predictions[i]
    predicted_ndvi = (predicted_ndvi - predicted_ndvi.min()) / (predicted_ndvi.max() - predicted_ndvi.min() + 1e-8)
    # Compute change map
    change_map = compute_change_map(previous_ndvi, predicted_ndvi, threshold=0.02)
    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(previous_ndvi, cmap='viridis')
    plt.title(f"Actual {future_years[i - 1] if i > 0 else 2024} (Normalized)")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_ndvi, cmap='viridis')
    plt.title(f"Predicted {year}")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(change_map, cmap='Reds')  # or 'hot'
    plt.title(f"Change Map ({future_years[i - 1] if i > 0 else 2024} → {year})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    # Save change map to output/predicted_images
    save_path = os.path.join(OUTPUT_DIR, f"change_map_{future_years[i - 1] if i > 0 else 2024}_to_{year}.png")
    plt.imsave(save_path, change_map.astype(float), cmap='Reds')
    previous_ndvi = predicted_ndvi.copy()
print(f"✅ All change maps saved in '{OUTPUT_DIR}' successfully!")


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image
# --- SETTINGS (inside 'output' folder of the project) ---
BASE_OUTPUT = os.path.join("..", "outputs")
change_map_folder = os.path.join(BASE_OUTPUT, "predicted_changemaps")
predicted_image_folder = os.path.join(BASE_OUTPUT, "predicted_images")  # NEW location for NDVI .npy files
signed_diff_folder = os.path.join(BASE_OUTPUT, "signed_diff_maps")
weighted_change_folder = os.path.join(BASE_OUTPUT, "weighted_change_maps")
classified_map_folder = os.path.join(BASE_OUTPUT, "classified_change_maps")
os.makedirs(signed_diff_folder, exist_ok=True)
os.makedirs(weighted_change_folder, exist_ok=True)
os.makedirs(classified_map_folder, exist_ok=True)
# Sorted list of change map filenames
change_map_files = sorted([
    f for f in os.listdir(change_map_folder)
    if f.endswith(".png") and "change_map" in f
])
# Initialize maps
cumulative_change = None
signed_diff = None
weighted_change = None
# Updated function to load NDVI predictions from predicted_images folder
def load_ndvi_prediction(year):
    path = os.path.join(predicted_image_folder, f"ndvi_{year}.npy")
    if not os.path.exists(path):
        print(f"NDVI file for {year} not found at {path}")
    return np.load(path) if os.path.exists(path) else None
# Start from 2024 (the last known year of input)
previous_ndvi = load_ndvi_prediction(2024)
for i, file in enumerate(change_map_files):
    year = 2025 + i
    print(f"Processing file: {file} for year {year}")
    next_ndvi = load_ndvi_prediction(year)
    # Binary Change Map
    path = os.path.join(change_map_folder, file)
    img = Image.open(path).convert("L")
    binary_map = np.array(img) > 128
    # Cumulative Change Map
    if cumulative_change is None:
        cumulative_change = np.zeros_like(binary_map, dtype=int)
    cumulative_change += binary_map.astype(int)
    # Signed Difference and Weighted Change Logic
    if previous_ndvi is not None and next_ndvi is not None:
        signed = next_ndvi - previous_ndvi
        signed_avg = np.mean(signed, axis=2)
        signed_diff = signed_avg if signed_diff is None else signed_diff + signed_avg
        np.save(os.path.join(signed_diff_folder, f"signed_diff_{year}.npy"), signed_avg)
        weighted = np.abs(signed_avg) * binary_map
        weighted_change = weighted if weighted_change is None else weighted_change + weighted
        np.save(os.path.join(weighted_change_folder, f"weighted_change_{year}.npy"), weighted)
    else:
        print(f"Skipping year {year} due to missing NDVI data.")
    previous_ndvi = next_ndvi
# --- Save and Plot Heatmaps ---
if cumulative_change is not None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cumulative_change, cmap='hot')
    plt.colorbar(label="Number of Times Deforested")
    plt.title("Cumulative Deforestation Heatmap (2024 → 2027)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(change_map_folder, "cumulative_deforestation_heatmap.png"))
    plt.show()


if signed_diff is not None:
    plt.figure(figsize=(8, 6))
    plt.imshow(signed_diff, cmap='seismic', norm=Normalize(vmin=-1, vmax=1))
    plt.colorbar(label="Signed NDVI Change")
    plt.title("Signed NDVI Change Map (2024 → 2027)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(signed_diff_folder, "signed_diff_heatmap.png"))
    plt.show()


if weighted_change is not None:
    plt.figure(figsize=(8, 6))
    plt.imshow(weighted_change, cmap='viridis')
    plt.colorbar(label="Weighted Change Intensity")
    plt.title("Weighted Deforestation Heatmap (2024 → 2027)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(weighted_change_folder, "weighted_change_heatmap.png"))
    plt.show()

#--- Classification Map ---
if signed_diff is not None:
    classified_map = np.zeros_like(signed_diff, dtype=np.uint8)
    classified_map[signed_diff > 0.05] = 2     # Improving
    classified_map[signed_diff < -0.05] = 1    # Degrading
    colors = ['gray', 'red', 'green']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    plt.figure(figsize=(8, 6))
    plt.imshow(classified_map, cmap=cmap, vmin=0, vmax=2)
    plt.title("Classified Change Map")
    labels = ["Stable", "Degrading", "Improving"]
    patches = [plt.matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
    plt.legend(handles=patches, loc='lower right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(classified_map_folder, "classified_change_map.png"))
    plt.show()
print(f"✅ All maps and heatmaps saved successfully!")


import csv
import os
import numpy as np
# Folder where you want to save the CSV
report_csv_path = os.path.join("..", "outputs", "predicted_changemaps", "change_summary_report.csv")
# Initialize previous NDVI
previous_ndvi = X[-1][-1]
previous_ndvi = (previous_ndvi - previous_ndvi.min()) / (previous_ndvi.max() - previous_ndvi.min() + 1e-8)
future_years = [2025, 2026, 2027]
threshold = 0.02  # Change threshold
# Create CSV file and write header
with open(report_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Year", "Degraded Pixels", "Improved Pixels", "Stable Pixels"])
    for i, year in enumerate(future_years):
        predicted_ndvi = future_predictions[i]
        predicted_ndvi = (predicted_ndvi - predicted_ndvi.min()) / (predicted_ndvi.max() - predicted_ndvi.min() + 1e-8)
        # Compute signed difference
        diff = predicted_ndvi - previous_ndvi
        diff_mean = np.mean(diff, axis=2)  # mean across channels
        degraded = np.sum(diff_mean <= -threshold)
        improved = np.sum(diff_mean >= threshold)
        stable = np.sum((diff_mean > -threshold) & (diff_mean < threshold))
        # Write row to CSV
        writer.writerow([year, degraded, improved, stable])
        # Update for next iteration
        previous_ndvi = predicted_ndvi.copy()
print(f"✅ CSV report saved to: {report_csv_path}")


import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
# --- SETTINGS ---
BASE_DIR = os.path.abspath("..")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "original_images_files")
# Step 1: Load Images in Time Series Order
def load_images(folder_path):
    image_paths = sorted(glob(os.path.join(folder_path, "*.png")))
    images = []
    years = []
    
    for path in image_paths:
        try:
            year = int(os.path.basename(path).split('_')[-1].split('.')[0])
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256))  # Resize images to 256x256 (or your preferred size)
            images.append(img)
            years.append(year)
        except:
            continue
    return np.array(images), years
images, years = load_images(DATA_DIR)
print(f"Total Years Loaded: {len(images)}")
# Step 2: Count Green Pixels in NDVI Image
def count_green_pixels(ndvi_image, green_threshold=0.4):
    """
    Count green pixels in an NDVI image.
    Assumes image shape (256, 256, 3) and uses the mean NDVI across channels.
    """
    avg_ndvi = np.mean(ndvi_image, axis=2)
    green_pixels = (avg_ndvi > green_threshold).sum()
    return green_pixels
# Step 3: Normalize and Count Green Pixels Year-wise
normalized_images = []
for img in images:
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize NDVI images
    normalized_images.append(img)
# Count green pixels year-wise
green_counts = [count_green_pixels(img) for img in normalized_images]
# Step 4: Calculate Deforestation Rate (percentage loss)
deforestation_rates = []
for i in range(1, len(green_counts)):
    change = green_counts[i-1] - green_counts[i]
    percent_loss = (change / green_counts[i-1]) * 100
    deforestation_rates.append(percent_loss)
# Step 5: Plot the Deforestation Rate from 2000 to 2024
plt.figure(figsize=(12, 5))
plt.plot(years[1:], deforestation_rates, marker='o', color='red', linewidth=2)
plt.title("Yearly Deforestation Rate (2000 → 2024)")
plt.xlabel("Year")
plt.ylabel("Deforestation Rate (%)")
plt.grid(True)
plt.xticks(years[1:])
plt.tight_layout()
# Save the graph in the output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, "deforestation_rate_2000_to_2024.png"))
plt.show()
print(f"✅ Deforestation Rate Graph saved successfully!")


import numpy as np
import matplotlib.pyplot as plt
# Count green pixels in an NDVI image
def count_green_pixels(ndvi_image, green_threshold=0.4):
    """
    Count green pixels in an NDVI image.
    Assumes image shape (256, 256, 3) and uses the mean NDVI across channels.
    """
    avg_ndvi = np.mean(ndvi_image, axis=2)
    green_pixels = (avg_ndvi > green_threshold).sum()
    return green_pixels
# Years and images (true NDVI for 2024 and predicted for 2025, 2026, 2027)
all_years = [2024, 2025, 2026, 2027]
all_ndvi_images = [X[-1][-1]] + future_predictions  # Assuming 'future_predictions' is already generated
# Normalize all NDVI images
normalized_images = []
for img in all_ndvi_images:
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize each image to range [0, 1]
    normalized_images.append(img)
# Count green pixels year-wise
green_counts = [count_green_pixels(img) for img in normalized_images]
# Calculate percentage change year-over-year (deforestation rate)
deforestation_rates = []
for i in range(1, len(green_counts)):
    change = green_counts[i-1] - green_counts[i]
    percent_loss = (change / green_counts[i-1]) * 100
    deforestation_rates.append(percent_loss)
# Plot Deforestation Rate Graph
plt.figure(figsize=(8, 5))
plt.plot(all_years[1:], deforestation_rates, marker='o', color='red', linewidth=2)
plt.title("Yearly Deforestation Rate")
plt.xlabel("Year")
plt.ylabel("Deforestation Rate (%)")
plt.grid(True)
plt.xticks(all_years[1:])
plt.tight_layout()
plt.show()


