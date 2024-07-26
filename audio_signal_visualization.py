# %%
pip install librosa

# %% [markdown]
# Show waveform

# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define the path to the audio file
audio_file_path = "./0638.mp3"

# Load the audio file
# audio_signal: the audio time series
# sample_rate: the sampling rate of the audio file
audio_signal, sample_rate = librosa.load(audio_file_path)

# Create a new figure for the plot
# figsize: (width, height) in inches
plt.figure(figsize=(10, 4))

# Plot the audio signal
# x-axis: Time (in samples)
# y-axis: Amplitude of the audio signal
plt.plot(audio_signal)

# Add a title to the plot
plt.title('Waveform of 0638.mp3')

# Label the x-axis
plt.xlabel('Time (samples)')

# Label the y-axis
plt.ylabel('Amplitude')

# Add a grid to the plot for better readability
plt.grid(True)

# Display the plot
plt.show()

# %% [markdown]
# Show Spectrum

# %%
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define the path to the audio file
audio_file_path = "./0638.mp3"

# Load the audio file
audio_signal, sample_rate = librosa.load(audio_file_path)

# Define the number of FFT components
n_fft = 2048

# Compute the Short-Time Fourier Transform (STFT) of the audio signal
# Taking only the first n_fft samples of the audio signal
# hop_length: number of audio frames between STFT columns (n_fft + 1 for this case)
stft_result = np.abs(librosa.stft(audio_signal[:n_fft], hop_length=n_fft + 1))

# Create a new figure for the plot
plt.figure(figsize=(10, 4))

# Plot the amplitude spectrum
plt.plot(stft_result)

# Add a title to the plot
plt.title('Spectrum of 0638.mp3')

# Label the x-axis
plt.xlabel('Frequency Bin')

# Label the y-axis
plt.ylabel('Amplitude')

# Add a grid to the plot for better readability
plt.grid(True)

# Display the plot
plt.show()


# %% [markdown]
# Show Spectogram
# 

# %%
# Import necessary libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file
file_path = r"./0638.mp3"
y, sr = librosa.load(file_path)

# Compute the Short-Time Fourier Transform (STFT) of the audio signal
D = np.abs(librosa.stft(y))

# Convert the amplitude spectrogram to dB-scaled spectrogram
DB = librosa.amplitude_to_db(D, ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()


# %% [markdown]
# Show Melspectogram
# 

# %%
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Define the path to the audio file
audio_file_path = "./0638.mp3"

# Load the audio file
audio_signal, sample_rate = librosa.load(audio_file_path)

# Define parameters for the Mel spectrogram
n_fft = 2048       # Number of FFT components
hop_length = 1024  # Number of audio frames between STFT columns

# Compute the Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)

# Convert the Mel spectrogram to decibel units
mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Create a new figure for the plot
plt.figure(figsize=(10, 4))

# Display the Mel spectrogram
# y_axis='mel': Mel scale on the y-axis
# fmax=8000: Maximum frequency to be displayed
# x_axis='time': Time on the x-axis
librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, hop_length=hop_length, y_axis='mel', fmax=8000, x_axis='time')

# Add a title to the plot
plt.title('Mel Spectrogram of 0638.mp3')

# Add a colorbar to the plot
# format='%+2.0f dB': Format of the colorbar labels
plt.colorbar(format='%+2.0f dB')

# Display the plot
plt.show()



