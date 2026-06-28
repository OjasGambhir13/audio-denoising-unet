import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os
import time
import torch
from inference import run_inference

def record_audio(duration=10, sample_rate=16000):
    print("\n" + "="*50)
    print(f"🎤 GET READY! Recording for {duration} seconds in...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)
        
    print("🔴 RECORDING NOW! (Talk and make some background noise)")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.float32)
    sd.wait() 
    print("✅ Recording complete.")
    print("="*50 + "\n")
    
    return audio_data.flatten(), sample_rate

def main():
    duration = 10         
    chunk_duration = 3    
    sample_rate = 16000
    
    os.makedirs('outputs', exist_ok=True)
    noisy_path = "outputs/live_10s_noisy.wav"
    clean_path = "outputs/live_10s_cleaned.wav"
    
    audio_data, sr = record_audio(duration, sample_rate)
    wav.write(noisy_path, sr, audio_data)
    
    print("🧠 Processing audio directly in RAM...")
    
    total_samples = len(audio_data)
    chunk_samples = int(chunk_duration * sample_rate)   
    hop_samples = int(chunk_samples / 2)                 
    
    if (total_samples - chunk_samples) % hop_samples != 0:
        remainder = (total_samples - chunk_samples) % hop_samples
        pad_len = hop_samples - remainder
        audio_data = np.pad(audio_data, (0, pad_len), mode='constant')
        total_samples = len(audio_data)
        
    output_audio = np.zeros(total_samples, dtype=np.float32)
    window_weights = np.zeros(total_samples, dtype=np.float32)
    
    window_func = np.ones(chunk_samples, dtype=np.float32)
    fade_len = hop_samples
    fade_in = np.linspace(0.0, 1.0, fade_len)
    fade_out = np.linspace(1.0, 0.0, fade_len)
    window_func[:fade_len] *= fade_in
    window_func[-fade_len:] *= fade_out

    start_idx = 0
    chunk_count = 0
    
    temp_mem_file = "outputs/ram_bridge.wav"
    
    while start_idx + chunk_samples <= total_samples:
        end_idx = start_idx + chunk_samples
        chunk_data = audio_data[start_idx:end_idx]
        
        wav.write(temp_mem_file, sample_rate, chunk_data)
        
        cleaned_chunk = run_inference(temp_mem_file, None)
        
        if len(cleaned_chunk) < chunk_samples:
            cleaned_chunk = np.pad(cleaned_chunk, (0, chunk_samples - len(cleaned_chunk)), mode='constant')
        elif len(cleaned_chunk) > chunk_samples:
            cleaned_chunk = cleaned_chunk[:chunk_samples]
            
        output_audio[start_idx:end_idx] += cleaned_chunk * window_func
        window_weights[start_idx:end_idx] += window_func
        
        chunk_count += 1
        print(f"📦 Processed window {chunk_count} ({start_idx/sample_rate:.1f}s to {end_idx/sample_rate:.1f}s)")
        start_idx += hop_samples

    safe_weights = np.where(window_weights > 0.001, window_weights, 1.0)
    final_audio = output_audio / safe_weights
    final_audio = final_audio[:int(duration * sample_rate)]
    
    wav.write(clean_path, sample_rate, final_audio)
    
    if os.path.exists(temp_mem_file): os.remove(temp_mem_file)
    
    print("\n🎉 SUCCESS!")
    print(f"1. Listen to your raw 10s recording: {noisy_path}")
    print(f"2. Listen to the AI cleaned version: {clean_path}")

if __name__ == '__main__':
    main()