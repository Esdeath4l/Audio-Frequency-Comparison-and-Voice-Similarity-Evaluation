import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# ===============================
# CONFIG
# ===============================
AUDIO_1_PATH = "first.wav"
AUDIO_2_PATH = "second.wav"
SAMPLE_RATE = 22050

# ===============================
# LOAD AUDIO
# ===============================
audio1, sr = librosa.load(AUDIO_1_PATH, sr=SAMPLE_RATE)
audio2, _  = librosa.load(AUDIO_2_PATH, sr=SAMPLE_RATE)

# Normalize
audio1 = librosa.util.normalize(audio1)
audio2 = librosa.util.normalize(audio2)

# ===============================
# FEATURE EXTRACTION
# ===============================

# MFCC
mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13)
mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13)

mfcc1_mean = np.mean(mfcc1, axis=1)
mfcc2_mean = np.mean(mfcc2, axis=1)

# Similarity
similarity_score = 1 - cosine(mfcc1_mean, mfcc2_mean)
similarity_percentage = similarity_score * 100

# STFT (Frequency domain)
spec1 = np.abs(librosa.stft(audio1))
spec2 = np.abs(librosa.stft(audio2))

spec1_mean = np.mean(spec1, axis=1)
spec2_mean = np.mean(spec2, axis=1)

# ===============================
# CONSOLE OUTPUT (PRO LEVEL)
# ===============================
print("\n" + "="*45)
print("🔊 ADVANCED AUDIO SIMILARITY ANALYSIS")
print("="*45)
print(f"🎧 Similarity Score      : {similarity_score:.4f}")
print(f"🎧 Similarity Percentage : {similarity_percentage:.2f}%")
print("="*45 + "\n")

# ===============================
# VISUAL DASHBOARD
# ===============================
plt.figure(figsize=(16, 12))

# ---- 1. Time Domain ----
plt.subplot(3, 1, 1)
plt.plot(audio1, label="Audio 1", alpha=0.7)
plt.plot(audio2, label="Audio 2", alpha=0.7)
plt.title("Time Domain Comparison")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(alpha=0.3)

# ---- 2. Frequency Domain ----
plt.subplot(3, 1, 2)
plt.plot(spec1_mean, label="Audio 1 Spectrum")
plt.plot(spec2_mean, label="Audio 2 Spectrum")
plt.title("Frequency Domain Comparison")
plt.xlabel("Frequency Bins")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(alpha=0.3)

# ---- Similarity Score on Graph ----
plt.text(
    0.02, 0.90,
    f"Similarity: {similarity_percentage:.2f}%",
    transform=plt.gca().transAxes,
    fontsize=13,
    bbox=dict(facecolor="white", alpha=0.85)
)

# ---- 3. MFCC Heatmap (Overlay Style) ----
plt.subplot(3, 2, 5)
librosa.display.specshow(mfcc1, x_axis="time", sr=sr, cmap="magma")
plt.colorbar()
plt.title("MFCC – Audio 1")

plt.subplot(3, 2, 6)
librosa.display.specshow(mfcc2, x_axis="time", sr=sr, cmap="magma")
plt.colorbar()
plt.title("MFCC – Audio 2")

plt.suptitle("Advanced Audio Frequency & Voice Similarity Dashboard", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
