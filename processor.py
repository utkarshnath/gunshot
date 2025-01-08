import librosa
import numpy as np
from scipy.signal import find_peaks, peak_prominences
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib
from scipy.stats import gaussian_kde
import serial
import time



# Convert frequency to Bark scale
def hz_to_bark(hz):
    return 13 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)

# Process a 5-second audio chunk and retrieve Bark spectrogram and peaks without buffering across segments
def process_audio_chunk(audio_chunk, sr, n_fft=2048, hop_length=512, prominence=25, buffer_ms=100):
    # Compute the magnitude spectrogram
    S = np.abs(librosa.stft(audio_chunk[:,0], n_fft=n_fft, hop_length=hop_length))
    # Retrieve peak information directly without segment buffering
    peaks_list = find_peaks_and_return_details(S, sr=sr, hop_length=hop_length, n_fft=n_fft, prominence=prominence)
    
    return peaks_list

# Find peaks in the spectrogram with a small buffer around each peak for averaging
def find_peaks_and_return_details(S, sr, hop_length=512, n_fft=2048, prominence=50, buffer_ms=100):
    decibel_spectrogram = librosa.amplitude_to_db(S, ref=np.max)
    decibel_over_time = np.mean(decibel_spectrogram, axis=0)
    peaks, properties = find_peaks(decibel_over_time, prominence=prominence)
    prominences = properties['prominences']
    #print(prominences)
    #prominences = peak_prominences(decibel_over_time, peaks)[0]

    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    bark_frequencies = hz_to_bark(frequencies)

    # Calculate how many frames correspond to the buffer duration (100 milliseconds)
    buffer_frames = int((buffer_ms / 1000) * (sr / hop_length))

    # Prepare list to hold the peak data
    peaks_list = []

    for peak_index, peak_time in zip(peaks, peak_times):
        # Define a small buffer around each peak to capture a range of decibel values
        start_index = max(0, peak_index - buffer_frames)
        end_index = min(S.shape[1], peak_index + buffer_frames)
        
        # Average decibel levels within this buffer region for all frequencies
        decibels_at_time = np.mean(S[:, start_index:end_index], axis=1)  # Average across buffer region
        
        peak_info = {
            'time': peak_time,
            'frequencies': bark_frequencies,  # Bark frequencies for the entire spectrogram
            'decibels': decibels_at_time      # Decibel values at the peak time (averaged within buffer)
        }
        peaks_list.append(peak_info)

    return peaks_list


def inference_all_models(path, data):
    knn_loaded = joblib.load(path + '/knn.joblib')
    pred_knn = knn_loaded.predict(data)
    #print("Prediction Nearest neighbour: ", pred_knn)

    clf_loaded = joblib.load(path + '/svm.joblib')
    pred_svm = clf_loaded.predict(data)
    #print("Prediction svm: ", pred_svm)

    pred = pred_knn + pred_svm
    #print(pred, max(pred))
    if(max(pred)==2):
        print("GunShot Detected")
        return 1
    return 0

def min_max_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def process_peaks(peaks_list):
    print()
    #print('#### Results with various models ####')
    all_decibels_normalized = []
    for i, peak in enumerate(peaks_list):
        decibels = peak['decibels']
        decibels_normalized = min_max_normalize(decibels)
        all_decibels_normalized.append(decibels_normalized)

    data = np.array(all_decibels_normalized)
    return inference_all_models('./model_v2', data)
    #print()
    #inference_all_models('./model_v1', data)
