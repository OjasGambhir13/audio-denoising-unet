import torch
import torchaudio
import matplotlib.pyplot as plt
import os

def calculate_snr(clean, noise, snr_db):
    """Calculates the scaling factor for noise to achieve a target SNR."""
    clean_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)

    if noise_power == 0:
        return noise
    
    snr_linear = 10 ** (snr_db / 10.0)
    target_noise_power = clean_power / snr_linear
    scale_factor = torch.sqrt(target_noise_power / noise_power)
    
    return noise * scale_factor

def process_audio_stft(waveform, n_fft=512, hop_length=256):
    """Computes the STFT and returns magnitude and phase."""
    window = torch.hann_window(n_fft).to(waveform.device)
    stft = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, 
                      window=window, return_complex=True)
    
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    
    magnitude = magnitude[:, :256, :]
    phase = phase[:, :256, :]
    return magnitude, phase

def inverse_stft(magnitude, phase, n_fft=512, hop_length=256):
    """Reconstructs the waveform using ISTFT."""
    pad = torch.zeros(magnitude.size(0), 1, magnitude.size(2)).to(magnitude.device)
    magnitude = torch.cat((magnitude, pad), dim=1)
    phase = torch.cat((phase, pad), dim=1)
    
    complex_stft = torch.polar(magnitude, phase)
    window = torch.hann_window(n_fft).to(magnitude.device)
    waveform = torch.istft(complex_stft, n_fft=n_fft, hop_length=hop_length, window=window)
    return waveform

def plot_spectrograms(noisy_mag, clean_mag, denoised_mag, filepath):
    """Saves a visual comparison of the spectrograms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    opts = dict(aspect='auto', origin='lower', cmap='magma')
    axes[0].imshow(torch.log1p(noisy_mag[0]).cpu().numpy(), **opts)
    axes[0].set_title('Noisy Input')
    axes[1].imshow(torch.log1p(denoised_mag[0]).cpu().numpy(), **opts)
    axes[1].set_title('Denoised Output')
    axes[2].imshow(torch.log1p(clean_mag[0]).cpu().numpy(), **opts)
    axes[2].set_title('Clean Target')
    
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()