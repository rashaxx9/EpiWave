"""
EpiWave Prediction API
Loads trained model and serves predictions for EEG files
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mne
import pywt
from pathlib import Path
import tempfile
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)


BASE_DIR = Path("EpiWave1/EpiWave_Model")
MODEL_PATH = BASE_DIR / "models" / "epiwave_multiclass_mobilenet_best.keras"

SELECTED_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7"]
WINDOW_SECONDS = 4
OVERLAP_SECONDS = 2
LOW_FREQ = 0.5
HIGH_FREQ = 40
NOTCH_FREQ = 60
IMAGE_SIZE = (224, 224)


print("Loading model...")
print(f"Looking for model at: {MODEL_PATH.absolute()}")
print(f"File exists: {MODEL_PATH.exists()}")

try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    model = None

def load_and_preprocess_edf(file_path):
    """Load EDF and return processed signal"""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    
   
    available_channels = [ch for ch in SELECTED_CHANNELS if ch in raw.ch_names]
    if not available_channels:
        
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch or 'FP' in ch or 'C' in ch]
        available_channels = eeg_channels[:3]
    
    raw.pick_channels(available_channels)
    raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    
    data = raw.get_data()
    sfreq = int(raw.info['sfreq'])
    
    # Average across channels
    signal = np.mean(data, axis=0)
    
    return signal, sfreq


def extract_segments(signal, sfreq):
    """Extract 4-second windows with 50% overlap"""
    window_samples = WINDOW_SECONDS * sfreq
    step_samples = int(window_samples * (1 - OVERLAP_SECONDS / WINDOW_SECONDS))
    
    segments = []
    for start in range(0, len(signal) - window_samples, step_samples):
        segment = signal[start:start + window_samples]
        start_sec = start / sfreq
        end_sec = start_sec + WINDOW_SECONDS
        segments.append({
            'segment': segment,
            'start_sec': start_sec,
            'end_sec': end_sec
        })
    
    return segments


def generate_cwt_array(segment, sfreq):
    """Generate CWT scalogram as numpy array"""
    frequencies = np.linspace(LOW_FREQ, HIGH_FREQ, 64)
    wavelet = 'morl'
    scales = pywt.central_frequency(wavelet) * sfreq / frequencies
    
    coefficients, _ = pywt.cwt(segment, scales, wavelet, sampling_period=1/sfreq)
    
    power = np.abs(coefficients)
    power = np.log1p(power)
    

    p_min, p_max = power.min(), power.max()
    if p_max > p_min:
        power = (power - p_min) / (p_max - p_min)
    else:
        power = np.zeros_like(power)
    
   
    from scipy.ndimage import zoom
    zoom_factor = IMAGE_SIZE[0] / power.shape[1]
    power = zoom(power, (1, zoom_factor), order=1)
    
    return power


def predict_segment(segment, sfreq):
    """Run prediction on a single EEG segment"""
    
    scalogram = generate_cwt_array(segment, sfreq)
    
    
    scalogram_3ch = np.stack([scalogram, scalogram, scalogram], axis=-1)
    input_array = np.expand_dims(scalogram_3ch, axis=0)
    
    
    probabilities = model.predict(input_array, verbose=0)[0]
    
    class_names = ['normal', 'preictal', 'seizure']
    predicted_idx = int(np.argmax(probabilities))
    
    return {
        'label': class_names[predicted_idx],
        'probabilities': {
            'normal': float(probabilities[0]),
            'preictal': float(probabilities[1]),
            'seizure': float(probabilities[2])
        }
    }


def generate_scalogram_image(segment, sfreq):
    """Generate base64 encoded scalogram image for display"""
    frequencies = np.linspace(LOW_FREQ, HIGH_FREQ, 64)
    wavelet = 'morl'
    scales = pywt.central_frequency(wavelet) * sfreq / frequencies
    
    coefficients, _ = pywt.cwt(segment, scales, wavelet, sampling_period=1/sfreq)
    
    power = np.abs(coefficients)
    power = np.log1p(power)
    
    p_min, p_max = power.min(), power.max()
    if p_max > p_min:
        power = (power - p_min) / (p_max - p_min)
    power = (power * 255).astype(np.uint8)
    
    img = Image.fromarray(power)
    img = img.resize(IMAGE_SIZE)
    img = img.convert('RGB')
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route('/analyze', methods=['POST'])
def analyze_eeg():
    """Main endpoint for EEG analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Load and process
        signal, sfreq = load_and_preprocess_edf(tmp_path)
        segments = extract_segments(signal, sfreq)
        
        if not segments:
            return jsonify({'error': 'No valid segments extracted'}), 400
        
        # Predict all segments
        results = []
        seizure_count = 0
        preictal_count = 0
        normal_count = 0
        max_seizure_prob = 0
        max_preictal_prob = 0
        
        for seg in segments:
            pred = predict_segment(seg['segment'], sfreq)
            results.append({
                'start_sec': seg['start_sec'],
                'end_sec': seg['end_sec'],
                'prediction': pred['label'],
                'probabilities': pred['probabilities']
            })
            
            if pred['label'] == 'seizure':
                seizure_count += 1
                max_seizure_prob = max(max_seizure_prob, pred['probabilities']['seizure'])
            elif pred['label'] == 'preictal':
                preictal_count += 1
                max_preictal_prob = max(max_preictal_prob, pred['probabilities']['preictal'])
            else:
                normal_count += 1
        
        # Determine overall diagnosis
        if seizure_count > 0:
            overall = 'seizure'
            confidence = max_seizure_prob
        elif preictal_count > 0:
            overall = 'preictal'
            confidence = max_preictal_prob
        else:
            overall = 'normal'
            # Average normal probability
            normal_probs = [r['probabilities']['normal'] for r in results]
            confidence = sum(normal_probs) / len(normal_probs)
        
        # Get sample scalogram image
        middle_segment = segments[len(segments)//2]['segment']
        scalogram_b64 = generate_scalogram_image(middle_segment, sfreq)
        
        return jsonify({
            'status': 'success',
            'overall_prediction': overall,
            'confidence': round(confidence * 100, 1),
            'segments_analyzed': len(results),
            'seizure_segments': seizure_count,
            'preictal_segments': preictal_count,
            'normal_segments': normal_count,
            'scalogram_image': scalogram_b64,
            'details': results[:5]  # First 5 segments
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        os.unlink(tmp_path)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model_loaded': True})


if __name__ == '__main__':
    print("=" * 50)
    print("EpiWave Prediction API")
    print("Model path:", MODEL_PATH)
    print("=" * 50)
    print("Starting Flask server on port 5000...") 
    app.run(host='0.0.0.0', port=5000, debug=False)