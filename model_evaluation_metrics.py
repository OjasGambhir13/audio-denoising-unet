import torch
import torchaudio
import numpy as np
from pystoi import stoi  # New import for STOI metric
from model import AudioUNet

def calculate_snr(clean_signal, tested_signal):
    """
    Calculates the Signal-to-Noise Ratio (SNR) in Decibels (dB).
    Higher is better.
    """
    noise = tested_signal - clean_signal
    
    signal_power = torch.sum(clean_signal ** 2)
    noise_power = torch.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
        
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def calculate_si_sdr(clean_signal, estimated_signal):
    """
    Calculates Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    This is the industry standard for speech separation. Higher is better.
    """
    clean_signal = clean_signal - torch.mean(clean_signal)
    estimated_signal = estimated_signal - torch.mean(estimated_signal)
    
    dot_product = torch.sum(estimated_signal * clean_signal)
    clean_energy = torch.sum(clean_signal ** 2) + 1e-8
    alpha = dot_product / clean_energy
    
    target = alpha * clean_signal
    error = estimated_signal - target
    
    si_sdr = 10 * torch.log10((torch.sum(target ** 2) + 1e-8) / (torch.sum(error ** 2) + 1e-8))
    return si_sdr.item()

def calculate_stoi(clean_signal, estimated_signal, sr=16000):
    """
    Calculates Short-Time Objective Intelligibility (STOI).
    Scale of 0.0 to 1.0 (1.0 is perfect human intelligibility).
    """
    clean_np = clean_signal.cpu().numpy()
    est_np = estimated_signal.cpu().numpy()
    return stoi(clean_np, est_np, sr, extended=False)

def calculate_rmse(clean_signal, estimated_signal):
    """
    Calculates Root Mean Square Error (RMSE) between the waveforms.
    Lower is better (closer to 0).
    """
    mse = torch.mean((clean_signal - estimated_signal) ** 2)
    return torch.sqrt(mse).item()

def generate_synthetic_test_pair(duration=3, sr=16000):
    """Generates a perfect clean signal and a heavily noisy version for testing."""
    t = torch.linspace(0, duration, sr * duration)
    
    clean = (torch.sin(2 * torch.pi * 300 * t) + 
             0.5 * torch.sin(2 * torch.pi * 700 * t) + 
             0.2 * torch.sin(2 * torch.pi * 1200 * t))
    
    envelope = torch.abs(torch.sin(2 * torch.pi * 1.5 * t))
    clean = clean * envelope
    
    noise = torch.randn_like(clean)
    
    clean = clean / torch.max(torch.abs(clean))
    noise = noise / torch.max(torch.abs(noise))
    noisy = clean + (noise * 0.8) # 80% noise!
    
    return clean.unsqueeze(0), noisy.unsqueeze(0)

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Booting up Evaluation Pipeline on {device}...")
    
    model = AudioUNet().to(device)
    try:
        model.load_state_dict(torch.load('models/best_unet.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("❌ Error: 'models/best_unet.pth' not found! Make sure you trained the model.")
        return
        
    model.eval()

    clean_audio, noisy_audio = generate_synthetic_test_pair()
    noisy_audio = noisy_audio.to(device)
    
    baseline_snr = calculate_snr(clean_audio.squeeze(), noisy_audio.cpu().squeeze())
    baseline_sisdr = calculate_si_sdr(clean_audio.squeeze(), noisy_audio.cpu().squeeze())

    print("📊 Extracting Spectrograms...")
    n_fft = 512
    hop_length = 256
    window = torch.hann_window(n_fft).to(device)
    
    stft_result = torch.stft(noisy_audio.squeeze(0), n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude = torch.abs(stft_result)
    phase = torch.angle(stft_result)

    mag_input = magnitude[:256, :].unsqueeze(0).unsqueeze(0)
    mag_max = torch.max(mag_input)
    if mag_max > 0:
        mag_input = mag_input / mag_max

    print("🧠 Passing through U-Net AI...")
    with torch.no_grad():
        denoised_mag = model(mag_input)
        
    if mag_max > 0:
        denoised_mag = denoised_mag * mag_max
        
    denoised_mag = torch.clamp(denoised_mag.squeeze(0).squeeze(0), min=0.0) 
    
    pad = torch.zeros(1, denoised_mag.size(1)).to(device)
    denoised_mag_padded = torch.cat((denoised_mag, pad), dim=0)
    
    complex_stft = torch.polar(denoised_mag_padded, phase)
    cleaned_waveform = torch.istft(complex_stft, n_fft=n_fft, hop_length=hop_length, window=window, length=noisy_audio.shape[1]).unsqueeze(0)

    cleaned_waveform = cleaned_waveform / torch.max(torch.abs(cleaned_waveform))
    cleaned_audio = cleaned_waveform.cpu().squeeze()
    clean_audio = clean_audio.squeeze()

    final_snr = calculate_snr(clean_audio, cleaned_audio)
    final_sisdr = calculate_si_sdr(clean_audio, cleaned_audio)
    
    baseline_stoi = calculate_stoi(clean_audio, noisy_audio.cpu().squeeze())
    final_stoi = calculate_stoi(clean_audio, cleaned_audio)
    
    baseline_rmse = calculate_rmse(clean_audio, noisy_audio.cpu().squeeze())
    final_rmse = calculate_rmse(clean_audio, cleaned_audio)

    print("\n" + "="*50)
    print(" 🎯 AUDIO DENOISING PERFORMANCE METRICS")
    print("="*50)
    print(f" Baseline Noisy SNR  : {baseline_snr:6.2f} dB")
    print(f" AI Cleaned SNR      : {final_snr:6.2f} dB")
    print(f" 🟢 SNR Improvement  : +{final_snr - baseline_snr:.2f} dB")
    print("-" * 50)
    print(f" Baseline SI-SDR     : {baseline_sisdr:6.2f} dB")
    print(f" AI Cleaned SI-SDR   : {final_sisdr:6.2f} dB")
    print(f" 🟢 SI-SDR Boost     : +{final_sisdr - baseline_sisdr:.2f} dB")
    print("-" * 50)
    print(f" Baseline STOI       : {baseline_stoi:6.4f} (Human Intelligibility)")
    print(f" AI Cleaned STOI     : {final_stoi:6.4f} ")
    print(f" 🟢 STOI Improvement : +{final_stoi - baseline_stoi:.4f}")
    print("-" * 50)
    print(f" Baseline RMSE Error : {baseline_rmse:6.4f} (Lower is better)")
    print(f" AI Cleaned RMSE     : {final_rmse:6.4f} ")
    print(f" 🟢 RMSE Reduction   : -{baseline_rmse - final_rmse:.4f}")
    print("="*50 + "\n")

if __name__ == '__main__':
    run_evaluation()