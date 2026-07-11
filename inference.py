import torch
import torchaudio
import argparse
import os
from model import AudioUNet

def run_inference(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading AI Model onto: {device}")
    
    # Load the U-Net model and its weights
    model = AudioUNet().to(device)
    model.load_state_dict(torch.load('models/best_unet.pth', map_location=device, weights_only=True))
    model.eval()

    waveform, sr = torchaudio.load(input_path)
    
    # Convert to 16kHz and mono to match the training data
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.to(device)

    chunk_size = 48000  # 3 seconds at 16kHz
    stride = 24000      # 1.5 seconds (50% overlap)
    n_fft = 512
    hop_length = 256
    window_fn = torch.hann_window(n_fft).to(device)
    
    total_samples = waveform.shape[1]
    
    # Pad if audio is shorter than 3 seconds
    if total_samples < chunk_size:
        waveform = torch.nn.functional.pad(waveform, (0, chunk_size - total_samples))
        total_samples = chunk_size
        
    output_waveform = torch.zeros(1, total_samples).to(device)
    overlap_count = torch.zeros(1, total_samples).to(device)
    crossfade_window = torch.hann_window(chunk_size).unsqueeze(0).to(device)

    print("Applying Sliding Window Denoising...")
    
    for start in range(0, total_samples, stride):
        end = start + chunk_size
        chunk = waveform[:, start:end]
        
        # Pad the very last chunk if it's cut off
        if chunk.shape[1] < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.shape[1]))
            
        # Extract the Spectrogram
        stft_result = torch.stft(chunk.squeeze(0), n_fft=n_fft, hop_length=hop_length, window=window_fn, return_complex=True)
        
        magnitude = torch.abs(stft_result)
        phase = torch.angle(stft_result)

        # Ensure shape matches the U-Net input (256 frequency bins)
        mag_input = magnitude[:256, :].unsqueeze(0).unsqueeze(0)

        # 1. Scale the input to 0-1 so the U-Net understands it
        mag_max = torch.max(mag_input)
        if mag_max > 0:
            mag_input = mag_input / mag_max

        # Run through the AI
        with torch.no_grad():
            denoised_mag = model(mag_input)
            
        # 2. Scale it back up to its original volume
        if mag_max > 0:
            denoised_mag = denoised_mag * mag_max
            
        # 3. Prevent any impossible negative frequencies
        denoised_mag = torch.clamp(denoised_mag, min=0.0)
            
        denoised_mag = denoised_mag.squeeze(0).squeeze(0) 
        
        # Add the 257th frequency bin back (padding with zeros)
        pad_freq = torch.zeros(1, denoised_mag.size(1)).to(device)
        denoised_mag_padded = torch.cat((denoised_mag, pad_freq), dim=0)
        
        # Combine the cleaned magnitude with the original phase
        complex_stft = torch.polar(denoised_mag_padded, phase)
        
        # ---> THIS IS THE FIX: Explicitly forcing length=chunk_size <---
        cleaned_chunk = torch.istft(
            complex_stft, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            window=window_fn, 
            length=chunk_size
        ).unsqueeze(0)

        actual_end = min(start + chunk_size, total_samples)
        chunk_len = actual_end - start
        
        output_waveform[:, start:actual_end] += (cleaned_chunk[:, :chunk_len] * crossfade_window[:, :chunk_len])
        overlap_count[:, start:actual_end] += crossfade_window[:, :chunk_len]

    # Normalize overlap volumes
    output_waveform = output_waveform / torch.clamp(overlap_count, min=1e-8)

    # Normalize the audio so it doesn't clip your speakers
    cleaned_waveform = output_waveform / torch.max(torch.abs(output_waveform))

    # Save the file to the outputs folder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, cleaned_waveform.cpu(), 16000)
    print(f"✅ Success! Cleaned audio saved to: {output_path}")

    return cleaned_waveform.squeeze().cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to your noisy WAV/FLAC file")
    parser.add_argument('--output', type=str, default="outputs/cleaned_audio.wav", help="Path to save the clean file")
    args = parser.parse_args()
    
    run_inference(args.input, args.output)