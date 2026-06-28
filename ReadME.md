Real-Time Audio Denoising U-Net 🎙️🧠

This repository contains a PyTorch-based Deep Learning pipeline designed to perform real-time speech enhancement and background noise suppression. It was built as an Engineering Project.

Overview

The system uses a Convolutional Neural Network (U-Net architecture) operating on audio spectrograms to isolate human speech from complex background noise. It supports sliding-window inference for processing arbitrary-length audio and includes a real-time live microphone demonstration.

Key Features

Custom U-Net Architecture: Deep autoencoder design with skip-connections and dynamic padding.

Spectrogram Processing: Short-Time Fourier Transform (STFT) matrix manipulation.

Sliding Window Inference: Processes infinite-length audio in overlapping 3-second chunks with linear crossfading to prevent boundary clipping.

Live Microphone Demo: Hooks directly into local hardware for real-time inference.

Project Structure

dataset.py - Custom PyTorch DataLoader with dynamic mixing and augmentation.

model.py - The U-Net Convolutional architecture.

train.py - Training loop with mixed-precision (AMP) and early stopping.

test.py - Evaluation script calculating SI-SDR and STOI metrics.

inference.py - Base script for processing single audio files.

live_10s_demo.py - Live microphone recording and sliding-window cleanup pipeline.

Getting Started

1. Install Dependencies

pip install torch torchaudio soundfile pystoi sounddevice scipy numpy tqdm


2. Prepare Datasets (For Training)

Place your clean .wav or .flac files in datasets/clean/ and your noise files in datasets/noise/.

3. Run the Live Demo

Ensure you have a trained model saved at models/best_unet.pth, then run:

python live_10s_demo.py
