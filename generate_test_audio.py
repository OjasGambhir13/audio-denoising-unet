import numpy as np
from scipy.io import wavfile

def generate_noisy_test_file():
    # Parameters matching your U-Net model
    sample_rate = 16000
    duration = 10  # 10 seconds
    t = np.linspace(0, duration, sample_rate * duration, False)

    print("Generating simulated human voice...")
    # 1. Generate "Speech-like" signal (Harmonics in the vocal range)
    f1, f2, f3 = 300, 700, 2500
    clean_signal = (
        np.sin(2 * np.pi * f1 * t) + 
        0.5 * np.sin(2 * np.pi * f2 * t) + 
        0.2 * np.sin(2 * np.pi * f3 * t)
    )
    
    # Add an envelope to mimic talking (pauses and syllables)
    envelope = np.abs(np.sin(2 * np.pi * 1.5 * t))  # ~1.5 "syllables" per second
    clean_signal = clean_signal * envelope

    print("Injecting background static noise...")
    # 2. Generate Background Noise (White Noise)
    noise = np.random.normal(0, 1, clean_signal.shape)

    # 3. Mix them together (Targeting a low SNR so the model has to work hard)
    clean_signal = clean_signal / np.max(np.abs(clean_signal))
    noise = noise / np.max(np.abs(noise))
    
    # Mix: 1 part signal, 0.7 parts noise
    noisy_signal = clean_signal + (noise * 0.7)

    # 4. Save to .wav format
    print("Saving to test_noisy_audio.wav...")
    # Scale to 16-bit PCM standard
    noisy_signal_int16 = np.int16(noisy_signal / np.max(np.abs(noisy_signal)) * 32767)
    
    wavfile.write("test_noisy_audio.wav", sample_rate, noisy_signal_int16)
    print("✅ Success! 'test_noisy_audio.wav' has been created in your folder.")

if __name__ == "__main__":
    generate_noisy_test_file()