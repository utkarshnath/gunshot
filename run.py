import sounddevice as sd
import numpy as np
from multiprocessing import Queue, Process
from scipy import signal
from processor import process_audio_chunk, process_peaks  # Import function to process each chunk
import time
from scipy.io.wavfile import write 
import os
import sys
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from scipy.signal import resample
import serial
import time

#ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Replace with your adapter's device name
#ser.flush()

# Suppress the specific warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Audio Configuration
samplerate = 48000
CHUNK = 24000  # 0.5 seconds if RATE is 48000 Hz
BUFFER_DURATION = 5  # Total duration of buffered audio for each processing cycle
BUFFER_CHUNKS = int(BUFFER_DURATION * (samplerate / CHUNK))  # Number of chunks needed to reach 5 seconds
input_gain_db = -12
downsample = 1
save_dir = ''

class Tee:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def set_gain_db(audiodata, gain_db):
    audiodata *= np.power(10, gain_db / 10)
    return np.array([1 if s > 1 else -1 if s < -1 else s for s in audiodata], dtype=np.float32)

def process_audio_data(audiodata):
    # Extract mono channels from input data
    ch1 = np.array(audiodata[::downsample, 0], dtype=np.float32)
    ch2 = np.array(audiodata[::downsample, 1], dtype=np.float32)
    ch1 = butter_highpass_filter(ch1, 10, samplerate)
    ch2 = butter_highpass_filter(ch2, 10, samplerate)
    ch1 = set_gain_db(ch1, input_gain_db)
    ch2 = set_gain_db(ch2, input_gain_db)
    return np.array([[ch1[i], ch2[i]] for i in range(len(ch1))], dtype=np.float32)

def record_audio(queue):
    print("Recording audio...")
    buffer = []

    def audio_callback(indata, frames, time, status):
        indata = resample(indata, int(frames * 16000 / samplerate))
        if status:
            print("[ERROR] Sounddevice status:", status)
        audio_np = process_audio_data(indata)
        buffer.append(audio_np)
        #print('buffer', audio_np.shape, len(buffer))
        if len(buffer) >= BUFFER_CHUNKS:
            # Combine chunks into a single 5-second buffer
            audio_chunk = np.concatenate(buffer, axis=0)
            queue.put(audio_chunk)
            print("[DEBUG] Added 2-second audio chunk to queue")
            buffer.clear()  # Reset buffer

    # Start the continuous recording with a callback
    with sd.InputStream(samplerate=samplerate, channels=2, callback=audio_callback, blocksize=CHUNK):
        while True:
            sd.sleep(1000)  # Keep the stream open indefinitely

def process_audio_chunks(queue):
    while True:
        if not queue.empty():
            audio_chunk = queue.get()
            print("[DEBUG] Retrieved a 2-second audio chunk from the queue")
            peaks = process_audio_chunk(audio_chunk, samplerate, buffer_ms=100)
            #print("[DEBUG] Detected Peaks:", peaks)
            if (len(peaks)>0):
               gunshots = process_peaks(peaks)
               #if gunshots:
                   #ser.write(b"1\n") 
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            write(save_dir + '/' + str(timestamp)+'_.wav', samplerate, audio_chunk)

if __name__ == "__main__":

    #global save_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = './output/' + str(timestamp)
    os.makedirs(save_dir)
    log_file = os.path.join(save_dir, "log.txt")
    sys.stdout = Tee(log_file)

    audio_queue = Queue()
    
    # Start recording process
    record_process = Process(target=record_audio, args=(audio_queue,))
    record_process.start()
    print("[DEBUG] Started audio recording process")
    
    # Start processing in parallel
    process_process = Process(target=process_audio_chunks, args=(audio_queue,))
    process_process.start()
    print("[DEBUG] Started audio processing process")
    
    record_process.join()
    process_process.join()

    #global ser
    #ser.close()

