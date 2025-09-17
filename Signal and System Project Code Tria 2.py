import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ----------- PAGE CONFIG & CUSTOM CSS -----------
st.set_page_config(
    page_title="Fourier Series Visualizer",
    page_icon="🎶",
    layout="wide"
)
st.markdown("""
<style>
body {
    background-color: #f5f5f7;
    color: #1d1d1f;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #1d1d1f;
    font-family: inherit;
}
.sidebar .sidebar-content {
    background: #f4f4f4;
}
.stButton>button {
    background-color: #0071e3;
    color: white;
    border-radius: 26px;
    padding: 6px 18px;
    border: none;
    font-size: 16px;
    transition: background-color 0.3s;
}
.stButton>button:hover {
    background-color: #005bb5;
}
[data-testid="stSidebar"] {
    background-color: #f5f5f7;
}
</style>
""", unsafe_allow_html=True)

# ----------- STYLISH HEADER -----------
st.markdown(
    """
    <h1 style='text-align: center; font-size: 3rem; font-weight: 700; margin-bottom: 0.3em;'>
        Fourier Series Visualization
    </h1>
    <p style='text-align: center; color: #444; font-size: 1.2rem;'>
        Explore square & sine wave decomposition interactively.<br>
        Minimalist interface, inspired by Apple design principles.
    </p>
    """,
    unsafe_allow_html=True
)

# ----------- SIDEBAR NAVIGATION (NO LOGO) -----------
st.sidebar.header("Customize")
func_name = st.sidebar.selectbox("Select waveform:", ["square", "sine"])
N = st.sidebar.slider("Fourier Terms (N)", 1, 20, 6)
st.sidebar.divider()
st.sidebar.write("""
*Instructions:*  
Choose a waveform and the number of Fourier terms  
to compare exact and approximated signals.
""")
st.sidebar.write("Made with ❤ using Streamlit")

# ------------- FOURIER SERIES FUNCTIONS -------------
def square_wave(t, T):
    t = np.mod(t, T)
    return np.where((t >= 0) & (t < T / 2), 1, -1)

def sine_wave(t, T):
    omega0 = 2 * np.pi / T
    return np.sin(omega0 * t)

def exponential_coefficients(func_name, n_max, T):
    coeffs = np.zeros(2 * n_max + 1, dtype=complex)
    if func_name == "square":
        for n in range(-n_max, n_max + 1):
            if n == 0:
                coeffs[n + n_max] = 0
            elif n % 2 != 0:
                coeffs[n + n_max] = -2j / (n * np.pi)
    elif func_name == "sine":
        coeffs[n_max + 1] = -0.5j
        coeffs[n_max - 1] = 0.5j
    return coeffs

def trigonometric_coefficients(func_name, n_max):
    a0 = 0
    a = np.zeros(n_max + 1)
    b = np.zeros(n_max + 1)
    if func_name == "square":
        for n in range(1, n_max + 1):
            if n % 2 != 0:
                b[n] = 4 / (n * np.pi)
    elif func_name == "sine":
        b[1] = 1
    return a0, a, b

def synthesize_exponential(t, coeffs, T):
    series = np.zeros_like(t, dtype=complex)
    n_max = (len(coeffs) - 1) // 2
    omega0 = 2 * np.pi / T
    for i, c in enumerate(coeffs):
        n = i - n_max
        series += c * np.exp(1j * n * omega0 * t)
    return np.real(series)

def synthesize_trigonometric(t, a0, a, b, T):
    series = a0 / 2 * np.ones_like(t)
    omega0 = 2 * np.pi / T
    for n in range(1, len(a)):
        series += a[n] * np.cos(n * omega0 * t) + b[n] * np.sin(n * omega0 * t)
    return series

# ------------- MAIN LAYOUT WITH COLUMNS -------------
T = 2 * np.pi
t = np.linspace(-3 * T, 3 * T, 1500)
if func_name == "square":
    orig = square_wave(t, T)
    title = "Square Wave"
else:
    orig = sine_wave(t, T)
    title = "Sine Wave"

coeffs_exp = exponential_coefficients(func_name, N, T)
approx_exp = synthesize_exponential(t, coeffs_exp, T)
a0, a, b = trigonometric_coefficients(func_name, N)
approx_tri = synthesize_trigonometric(t, a0, a, b, T)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"<h2 style='font-weight: 600;'>{title} Fourier Series Approximation</h2>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, orig, 'k--', label="Original Signal", linewidth=2)
    ax.plot(t, approx_exp, color='#0071e3', label=f"Exponential (N={N})", linewidth=2)
    ax.plot(t, approx_tri, color='#ff9500', label=f"Trigonometric (N={N})", linewidth=2)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)
    st.pyplot(fig)
    st.write("""
        **Explore how both Fourier series types together reconstruct the waveform.  
        Try changing N for sharper approximations!**
    """)

with col2:
    st.markdown("<h4>Coefficient Spectra</h4>", unsafe_allow_html=True)
    n_vals = np.arange(-N, N+1)
    coeffs_spectrum = exponential_coefficients(func_name, N, T)
    figM, axM = plt.subplots(figsize=(4,2.4))
    axM.stem(n_vals, np.abs(coeffs_spectrum), basefmt=" ", linefmt='#9476dd', markerfmt='D')
    axM.set_title("Magnitude |cₙ|", fontsize=11)
    axM.set_xlabel("n", fontsize=9)
    axM.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(figM)
    figT, axT = plt.subplots(figsize=(4,2.4))
    barw = 0.35
    indices = np.arange(len(a))
    axT.bar(indices - barw/2, a, width=barw, alpha=0.8, label="aₙ (cos)", color='#53c7f3')
    axT.bar(indices + barw/2, b, width=barw, alpha=0.8, label="bₙ (sin)", color='#fa8231')
    axT.set_title("Trig. Coefficients", fontsize=11)
    axT.set_xlabel("n", fontsize=9)
    axT.legend(fontsize=9)
    axT.grid(True, linestyle='--', alpha=0.4)
    st.pyplot(figT)
    st.write("""
        <span style='font-size:0.99em; color:#666;'>
        Both types of coefficients offer insight into spectral content of the signal.
        </span>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<p style='text-align:center; color:#aaa; font-size: 0.9rem; margin-top:4em;'>
Designed by College Student | Inspired by Apple | Built with Streamlit
</p>
""", unsafe_allow_html=True)