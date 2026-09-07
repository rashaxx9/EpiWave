
"""
EpiWave Multi-Class EEG Model
Classes:
0 = normal
1 = preictal
2 = seizure

Preictal = 2 minutes before seizure onset.

Pipeline:
EDF -> filtering -> 4-second windows -> CWT images -> MobileNetV2 -> normal/preictal/seizure

Run on Windows:
python epiwave_multiclass_model.py
"""

import random
import shutil
from pathlib import Path

import mne
import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



# 1. CONFIG


BASE_DIR = Path("EpiWave_Model")
RAW_DATA_DIR = BASE_DIR / "data_raw"
IMAGE_DIR = BASE_DIR / "cwt_images"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7"]

WINDOW_SECONDS = 4
OVERLAP_SECONDS = 2
PREICTAL_SECONDS = 120

LOW_FREQ = 0.5
HIGH_FREQ = 40
NOTCH_FREQ = 60
IMAGE_SIZE = (224, 224)

MAX_NORMAL_PER_FILE = 40
MAX_PREICTAL_PER_FILE = 40
MAX_SEIZURE_PER_FILE = 150

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)



# 2. SEIZURE TIMES
# Add more CHB-MIT seizure files here if needed.

SEIZURE_ANNOTATIONS = {
    "chb01_03.edf": [(2996, 3036)],
    "chb01_04.edf": [(1467, 1494)],
    "chb01_15.edf": [(1732, 1772)],
    "chb01_16.edf": [(1015, 1066)],
    "chb01_18.edf": [(1720, 1810)],
    "chb01_21.edf": [(327, 420)],
    "chb01_26.edf": [(1862, 1963)],
}


# 3. DATA PREPARATION


def reset_image_folders():
    if IMAGE_DIR.exists():
        shutil.rmtree(IMAGE_DIR)

    for split_name in ["train", "val", "test"]:
        for class_name in ["normal", "preictal", "seizure"]:
            folder = IMAGE_DIR / split_name / class_name
            folder.mkdir(parents=True, exist_ok=True)


def get_window_label(start_sec, end_sec, seizure_intervals):
    # First priority: actual seizure
    for seizure_start, seizure_end in seizure_intervals:
        if start_sec < seizure_end and end_sec > seizure_start:
            return "seizure"

    # Second priority: preictal period before seizure
    for seizure_start, _ in seizure_intervals:
        preictal_start = max(0, seizure_start - PREICTAL_SECONDS)
        preictal_end = seizure_start

        if start_sec < preictal_end and end_sec > preictal_start:
            return "preictal"

    return "normal"


def load_and_preprocess_edf(edf_path):
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    available_channels = []
    for channel in SELECTED_CHANNELS:
        if channel in raw.ch_names:
            available_channels.append(channel)

    if len(available_channels) == 0:
        raise ValueError(
            "None of the selected channels were found in "
            + edf_path.name
            + ". Available channels: "
            + str(raw.ch_names)
        )

    raw.pick_channels(available_channels)
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)

    data = raw.get_data()
    sfreq = int(raw.info["sfreq"])

    signal = np.mean(data, axis=0)
    return signal, sfreq


def generate_cwt_image(segment, sfreq, save_path):
    frequencies = np.linspace(LOW_FREQ, HIGH_FREQ, 64)
    wavelet = "morl"
    scales = pywt.central_frequency(wavelet) * sfreq / frequencies

    coefficients, _ = pywt.cwt(
        segment,
        scales,
        wavelet,
        sampling_period=1 / sfreq
    )

    power = np.abs(coefficients)
    power = np.log1p(power)

    power_min = power.min()
    power_max = power.max()
    power = (power - power_min) / (power_max - power_min + 1e-8)
    power = (power * 255).astype(np.uint8)

    image = Image.fromarray(power)
    image = image.resize(IMAGE_SIZE)
    image = image.convert("RGB")
    image.save(save_path)


def generate_images_from_edf(edf_path):
    seizure_intervals = SEIZURE_ANNOTATIONS.get(edf_path.name, [])
    print("Processing", edf_path.name, "| seizures:", seizure_intervals)

    signal, sfreq = load_and_preprocess_edf(edf_path)

    window_size = WINDOW_SECONDS * sfreq
    step_size = (WINDOW_SECONDS - OVERLAP_SECONDS) * sfreq
    total_samples = len(signal)

    buckets = {
        "normal": [],
        "preictal": [],
        "seizure": []
    }

    for start_sample in range(0, total_samples - window_size, step_size):
        end_sample = start_sample + window_size
        start_sec = start_sample / sfreq
        end_sec = end_sample / sfreq

        segment = signal[start_sample:end_sample]
        label = get_window_label(start_sec, end_sec, seizure_intervals)

        buckets[label].append((segment, start_sec))

    random.shuffle(buckets["normal"])
    random.shuffle(buckets["preictal"])
    random.shuffle(buckets["seizure"])

    buckets["normal"] = buckets["normal"][:MAX_NORMAL_PER_FILE]
    buckets["preictal"] = buckets["preictal"][:MAX_PREICTAL_PER_FILE]
    buckets["seizure"] = buckets["seizure"][:MAX_SEIZURE_PER_FILE]

    temp_dir = IMAGE_DIR / "temp"

    for class_name in ["normal", "preictal", "seizure"]:
        class_folder = temp_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

    for class_name, samples in buckets.items():
        for index, (segment, start_sec) in enumerate(samples):
            file_name = edf_path.stem + "_" + class_name + "_" + str(index) + "_" + str(int(start_sec)) + ".png"
            save_path = temp_dir / class_name / file_name
            generate_cwt_image(segment, sfreq, save_path)

    print(
        "Saved",
        len(buckets["normal"]),
        "normal,",
        len(buckets["preictal"]),
        "preictal,",
        len(buckets["seizure"]),
        "seizure images from",
        edf_path.name
    )


def split_temp_images():
    temp_dir = IMAGE_DIR / "temp"

    for class_name in ["normal", "preictal", "seizure"]:
        files = list((temp_dir / class_name).glob("*.png"))
        random.shuffle(files)

        if len(files) < 3:
            print("WARNING: Not enough", class_name, "samples to split. Found", len(files))
            continue

        train_files, remaining_files = train_test_split(
            files,
            test_size=0.30,
            random_state=RANDOM_SEED
        )

        val_files, test_files = train_test_split(
            remaining_files,
            test_size=0.50,
            random_state=RANDOM_SEED
        )

        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]:
            for file in split_files:
                destination = IMAGE_DIR / split_name / class_name / file.name
                shutil.copy(file, destination)

        print(
            class_name + ":",
            "train=" + str(len(train_files)) + ",",
            "val=" + str(len(val_files)) + ",",
            "test=" + str(len(test_files))
        )


def build_dataset_from_raw():
    reset_image_folders()

    edf_files = list(RAW_DATA_DIR.rglob("*.edf"))

    if len(edf_files) == 0:
        raise FileNotFoundError("No EDF files found in: " + str(RAW_DATA_DIR))

    for edf_path in edf_files:
        generate_images_from_edf(edf_path)

    split_temp_images()



# 4. MODEL TRAINING


def build_mobilenet_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    output = Dense(3, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    return model


def train_model():
    train_dir = IMAGE_DIR / "train"
    val_dir = IMAGE_DIR / "val"

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode="categorical",
        shuffle=False
    )

    print("Class mapping:", train_generator.class_indices)

    model = build_mobilenet_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2
        ),
        ModelCheckpoint(
            MODEL_DIR / "epiwave_multiclass_mobilenet_best.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    class_weight = {
        0: 2.0,  # normal - increased to reduce false preictal alerts
        1: 3.0,  # preictal - reduced to prevent over-prediction
        2: 8.0   # seizure - kept high because missing seizures is critical
    }

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weight
    )

    model.save(MODEL_DIR / "epiwave_multiclass_mobilenet_final.keras")
    print("Model saved successfully.")

    return model, history



# 5. EVALUATION


def evaluate_model(model_path=None):
    if model_path is None:
        model_path = MODEL_DIR / "epiwave_multiclass_mobilenet_best.keras"

    model = tf.keras.models.load_model(model_path)

    test_dir = IMAGE_DIR / "test"
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode="categorical",
        shuffle=False
    )

    probabilities = model.predict(test_generator)
    predictions = np.argmax(probabilities, axis=1)
    true_labels = test_generator.classes

    class_names = ["normal", "preictal", "seizure"]

    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(true_labels, predictions)

    info = "\nMulti-class model: normal / preictal / seizure\n"
    info = info + "Preictal windows are defined as the 2 minutes before seizure onset.\n"

    print(report)
    print(info)
    print("Confusion Matrix:")
    print(cm)

    report_path = RESULTS_DIR / "multiclass_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
        file.write(info)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Multi-Class Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1, 2], class_names)
    plt.yticks([0, 1, 2], class_names)

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, cm[row, col], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "multiclass_confusion_matrix.png")
    plt.close()

    print("Results saved in", RESULTS_DIR)



# 6. SINGLE SEGMENT PREDICTION


def predict_single_segment(segment, sfreq, model_path=None):
    if model_path is None:
        model_path = MODEL_DIR / "epiwave_multiclass_mobilenet_best.keras"

    temp_path = RESULTS_DIR / "temp_multiclass_prediction.png"
    generate_cwt_image(segment, sfreq, temp_path)

    model = tf.keras.models.load_model(model_path)

    image = Image.open(temp_path).resize(IMAGE_SIZE).convert("RGB")
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array)[0]
    class_names = ["normal", "preictal", "seizure"]
    predicted_index = int(np.argmax(probabilities))

    return {
        "label": class_names[predicted_index],
        "normal_probability": float(probabilities[0]),
        "preictal_probability": float(probabilities[1]),
        "seizure_probability": float(probabilities[2])
    }



# 7. MAIN


if __name__ == "__main__":
    print("Step 1: Generating CWT images...")
    build_dataset_from_raw()

    print("Step 2: Training MobileNetV2 multi-class model...")
    trained_model, training_history = train_model()

    print("Step 3: Evaluating model...")
    evaluate_model()

    print("Done. Multi-class EpiWave model is ready.")
