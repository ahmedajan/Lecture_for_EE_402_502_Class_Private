# %%
pip install librosa matplotlib numpy scipy


# %% [markdown]
# Noise Reduction
# 

# %%
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
file_path = r"./0638.mp3"
y, sr = librosa.load(file_path)

# Plot original waveform
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Noise reduction using more aggressive spectral gating
def reduce_noise(y, sr, noise_reduction_factor=0.1, n_iter=3):
    for _ in range(n_iter):
        # Compute the Short-Time Fourier Transform (STFT) of the audio signal
        stft = librosa.stft(y)
        stft_magnitude, stft_phase = librosa.magphase(stft)

        # Compute the mean magnitude spectrum of the noise
        noise_magnitude = np.mean(stft_magnitude[:, :int(sr * 0.1)], axis=1)
        
        # Create a mask that attenuates regions below the noise threshold
        mask = stft_magnitude > noise_reduction_factor * noise_magnitude[:, np.newaxis]
        
        # Apply the mask to the magnitude spectrogram
        stft_magnitude_denoised = stft_magnitude * mask
        
        # Inverse STFT to get the denoised signal
        stft_denoised = stft_magnitude_denoised * stft_phase
        y = librosa.istft(stft_denoised)
    
    return y

# Apply aggressive noise reduction
y_denoised = reduce_noise(y, sr, noise_reduction_factor=10, n_iter=10)

# Plot denoised waveform
plt.subplot(2, 1, 2)
librosa.display.waveshow(y_denoised, sr=sr)
plt.title('Denoised Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()


# %% [markdown]
# MFCC extraction
# 

# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
file_path = r"./0638.mp3"
y, sr = librosa.load(file_path)

# Extract MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Plot the MFCCs
plt.figure(figsize=(10, 6))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.show()



