import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import io
from scipy.fft import fft, fftfreq

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🎶 Fourier Series Visualization 🎶", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap" rel="stylesheet">
<style>
body { background-color: #0f0f0f; font-family: 'Orbitron', sans-serif; }
h1, h2, h3 { color: #00ffe1; }
.stTabs [data-baseweb="tab"] { background-color: #1a1a1a; border-radius: 8px; padding: 10px; margin-right: 4px; }
.stTabs [aria-selected="true"] { background-color: #00ffe1; color: black; }
.glass-panel { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 20px; backdrop-filter: blur(10px); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎶 Fourier Series Visualization 🎶</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888;'>Record audio, explore Fourier decomposition & manipulate sound</p>", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "audio_data" not in st.session_state:
    st.session_state["audio_data"] = None
if "fs" not in st.session_state:
    st.session_state["fs"] = 44100
if "is_recording" not in st.session_state:
    st.session_state["is_recording"] = False
if "modifiers" not in st.session_state:
    st.session_state["modifiers"] = {"amp": 1.0, "freq": 1.0, "pitch": 1.0}

# ---------- FUNCTIONS ----------
def record_audio(duration=5, fs=44100):
    """Record audio for a given duration"""
    st.info("🎙️ Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(recording)

def play_audio(data, fs):
    """Return WAV bytes for streamlit audio player"""
    bio = io.BytesIO()
    sf.write(bio, data, fs, format="WAV")
    return bio.getvalue()

def trig_coeff(signal, n_max, T, t):
    a0 = np.mean(signal)
    a = np.zeros(n_max + 1)
    b = np.zeros(n_max + 1)
    w0 = 2 * np.pi / T
    for n in range(1, n_max + 1):
        a[n] = np.sum(signal * np.cos(n * w0 * t)) / len(t)
        b[n] = np.sum(signal * np.sin(n * w0 * t)) / len(t)
    return a0, a, b

def trig_recon(t, a0, a, b, T):
    w0 = 2 * np.pi / T
    s = a0 / 2 * np.ones_like(t)
    for n in range(1, len(a)):
        s += a[n] * np.cos(n * w0 * t) + b[n] * np.sin(n * w0 * t)
    return s

def exp_coeff(signal, n_max, T, t):
    coeffs = np.zeros(2 * n_max + 1, dtype=complex)
    w0 = 2 * np.pi / T
    for i, n in enumerate(range(-n_max, n_max + 1)):
        coeffs[i] = np.sum(signal * np.exp(-1j * n * w0 * t)) / len(t)
    return coeffs

def exp_recon(t, coeffs, T):
    s = np.zeros_like(t, dtype=complex)
    n_max = (len(coeffs) - 1) // 2
    w0 = 2 * np.pi / T
    for i, c in enumerate(coeffs):
        n = i - n_max
        s += c * np.exp(1j * n * w0 * t)
    return np.real(s)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["🎤 Recorder", "📊 Fourier Analysis"])

# ---------- TAB 1: RECORDER ----------
with tab1:
    st.subheader("🎙️ Audio Recorder")

    dur = st.slider("Recording Duration (seconds)", 1, 10, 5)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶ Start Recording"):
            audio = record_audio(duration=dur, fs=st.session_state["fs"])
            st.session_state["audio_data"] = audio
            st.success("✅ Recording finished")
    with col2:
        if st.button("⏸ Pause Recording"):
            st.warning("Pause not available in this mode (try WebRTC).")
    with col3:
        if st.button("⏹ Stop Recording"):
            st.session_state["is_recording"] = False
            st.info("Recording stopped.")

    if st.session_state["audio_data"] is not None:
        st.audio(play_audio(st.session_state["audio_data"], st.session_state["fs"]), format="audio/wav")

        st.markdown("### 🎚️ Modifiers")
        st.session_state["modifiers"]["amp"] = st.slider("Amplitude", 0.1, 3.0, 1.0, 0.1)
        st.session_state["modifiers"]["freq"] = st.slider("Frequency Scale", 0.5, 2.0, 1.0, 0.1)
        st.session_state["modifiers"]["pitch"] = st.slider("Pitch Shift", 0.5, 2.0, 1.0, 0.1)

# ---------- TAB 2: FOURIER ----------
with tab2:
    st.subheader("📊 Fourier Analysis of Recorded Audio")

    if st.session_state["audio_data"] is None:
        st.info("🎤 Record audio first in 'Recorder' tab.")
    else:
        data = st.session_state["audio_data"]  # ORIGINAL audio only
        fs = st.session_state["fs"]
        T = len(data) / fs
        t = np.linspace(0, T, len(data), endpoint=False)

        N = st.slider("Fourier Terms (N)", 1, 50, 20)

        # --- Exponential Fourier ---
        st.markdown("### 🧠 Exponential Fourier Series")
        coeffs = exp_coeff(data, N, T, t)

        figR, axR = plt.subplots(figsize=(6, 2.5))
        n_vals = np.arange(-N, N + 1)
        axR.stem(n_vals, np.real(coeffs), linefmt='#00ffe1', markerfmt='o', basefmt=" ")
        axR.set_title("Real Part Re(cₙ)")
        st.pyplot(figR)

        figI, axI = plt.subplots(figsize=(6, 2.5))
        axI.stem(n_vals, np.imag(coeffs), linefmt='#ff4081', markerfmt='o', basefmt=" ")
        axI.set_title("Imaginary Part Im(cₙ)")
        st.pyplot(figI)

        # --- Trigonometric Fourier ---
        st.markdown("### 📉 Trigonometric Fourier Series")
        a0, a, b = trig_coeff(data, N, T, t)
        recon = trig_recon(t, a0, a, b, T)

        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(t[:1000], data[:1000], 'k--', label="Original")
        ax2.plot(t[:1000], recon[:1000], color="#ff4081", linestyle="--", label=f"Trig Recon (N={N})")
        ax2.legend()
        st.pyplot(fig2)
