import torch
import torchaudio
import argparse
import os
from model import AudioUNet

def run_inference(input_path, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading AI Model onto: {device}")
    
    
    model = AudioUNet().to(device)
    model.load_state_dict(torch.load('models/best_unet.pth', weights_only=True))
    model.eval()

    waveform, sr = torchaudio.load(input_path)
    
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    n_fft = 512
    hop_length = 256
    window = torch.hann_window(n_fft).to(device)
    waveform = waveform.to(device)
    
    stft_result = torch.stft(waveform.squeeze(0), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    
    magnitude = torch.abs(stft_result)
    phase = torch.angle(stft_result)

    mag_input = magnitude[:256, :].unsqueeze(0).unsqueeze(0)

    print("Denoising...")
    with torch.no_grad():
        denoised_mag = model(mag_input)
        
    denoised_mag = denoised_mag.squeeze(0).squeeze(0) 
    
    pad = torch.zeros(1, denoised_mag.size(1)).to(device)
    denoised_mag_padded = torch.cat((denoised_mag, pad), dim=0)
    
    complex_stft = torch.polar(denoised_mag_padded, phase)
    cleaned_waveform = torch.istft(complex_stft, n_fft=n_fft, hop_length=hop_length, window=window).unsqueeze(0)

    cleaned_waveform = cleaned_waveform / torch.max(torch.abs(cleaned_waveform))

    return cleaned_waveform.squeeze().cpu().numpy()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to your noisy WAV/FLAC file")
    parser.add_argument('--output', type=str, default="outputs/cleaned_audio.wav", help="Path to save the clean file")
    args = parser.parse_args()
    
    run_inference(args.input, args.output)