"""Análisis reproducible de filtros digitales sobre vibración de un motor simulado.

Ejecuta: python analisis_vibracion.py
Los gráficos y métricas quedan dentro de las carpetas graficas/ y resultados/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby2, firwin, freqz, sosfiltfilt, filtfilt, welch

# Carpetas relativas al archivo para que el proyecto funcione desde cualquier ruta.
ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "graficas"
RESULTS = ROOT / "resultados"
PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

# Parámetros de adquisición sintética.
FS = 1000.0  # Hz
DURATION = 12.0  # segundos
SEED = 13112026
rng = np.random.default_rng(SEED)
t = np.arange(0, DURATION, 1 / FS)

# Vibración objetivo: componentes de giro y resonancias breves por impactos.
rotational = (
    0.45 * np.sin(2 * np.pi * 24 * t + 0.2)
    + 0.18 * np.sin(2 * np.pi * 48 * t - 0.4)
    + 0.07 * np.sin(2 * np.pi * 72 * t + 0.7)
)
impact_train = np.zeros_like(t)
impact_times = np.arange(0.35, DURATION, 0.12)
for center in impact_times:
    # Excitación breve que produce una resonancia amortiguada alrededor de 120 Hz.
    start = int(center * FS)
    length = min(int(0.09 * FS), len(t) - start)
    local_t = np.arange(length) / FS
    impact_train[start:start + length] += 0.38 * np.exp(-local_t / 0.025) * np.sin(2 * np.pi * 120 * local_t)
clean = rotational + impact_train

# Perturbaciones independientes: bamboleo lento, interferencia tonal y ruido blanco.
drift = 0.16 * np.sin(2 * np.pi * 1.2 * t)
interference = 0.14 * np.sin(2 * np.pi * 230 * t + 0.5)
white_noise = rng.normal(0, 0.095, size=t.size)
observed = clean + drift + interference + white_noise

# Filtros con diferentes respuestas: IIR Butterworth, IIR Chebyshev II y FIR Kaiser.
sos_low = butter(5, 65, btype="lowpass", fs=FS, output="sos")
sos_high = cheby2(5, 35, 65, btype="highpass", fs=FS, output="sos")
taps_band = firwin(501, [12, 160], pass_zero=False, fs=FS, window=("kaiser", 8.0))
filtered = {
    "Pasa bajos Butterworth": sosfiltfilt(sos_low, observed),
    "Pasa altos Chebyshev II": sosfiltfilt(sos_high, observed),
    "Pasa bandas FIR Kaiser": filtfilt(taps_band, [1.0], observed),
}

# Se excluyen los extremos para reducir la influencia del transitorio de borde.
mask = (t >= 1.0) & (t <= 11.0)
def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))
def snr_db(reference, estimate):
    error = reference - estimate
    return float(10 * np.log10(np.mean(reference ** 2) / np.mean(error ** 2)))

rows = []
for label, output in filtered.items():
    rows.append({
        "filtro": label,
        "RMS_salida": rms(output[mask]),
        "SNR_dB": snr_db(clean[mask], output[mask]),
        "mejora_SNR_dB": snr_db(clean[mask], output[mask]) - snr_db(clean[mask], observed[mask]),
        "RMSE": rms(clean[mask] - output[mask]),
    })
metrics = pd.DataFrame(rows)
metrics.to_csv(RESULTS / "metricas_filtros.csv", index=False, float_format="%.6f")
summary = {
    "frecuencia_muestreo_Hz": FS,
    "duracion_s": DURATION,
    "semilla": SEED,
    "SNR_entrada_dB": snr_db(clean[mask], observed[mask]),
    "metricas": rows,
}
(RESULTS / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

# Estilo gráfico consistente y legible.
plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25})
colors = {"Pasa bajos Butterworth": "#d26a4a", "Pasa altos Chebyshev II": "#7b6aa6", "Pasa bandas FIR Kaiser": "#2f8177"}

# 1. Entrada y espectro para ubicar el ruido junto a las frecuencias útiles.
fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), constrained_layout=True)
axes[0].plot(t, observed, color="#29434e", lw=0.65, label="Vibración observada")
axes[0].plot(t, clean, color="#d26a4a", lw=0.8, alpha=0.85, label="Referencia útil")
axes[0].set(xlim=(0, 1.5), xlabel="Tiempo (s)", ylabel="Amplitud relativa", title="Registro de vibración en el arranque")
axes[0].legend(ncol=2, frameon=False)
f, pxx = welch(observed, fs=FS, nperseg=4096)
axes[1].plot(f, 10 * np.log10(pxx + 1e-14), color="#29434e", lw=1)
axes[1].axvline(1.2, color="#d26a4a", ls="--", lw=1, label="Deriva 1.2 Hz")
axes[1].axvline(120, color="#2f8177", ls="--", lw=1, label="Resonancia 120 Hz")
axes[1].axvline(230, color="#7b6aa6", ls="--", lw=1, label="Interferencia 230 Hz")
axes[1].set(xlim=(0, 300), xlabel="Frecuencia (Hz)", ylabel="PSD (dB/Hz)", title="Densidad espectral de la entrada")
axes[1].legend(ncol=3, frameon=False)
fig.savefig(PLOTS / "entrada_motor.png", bbox_inches="tight")
plt.close(fig)

# 2. Respuesta de magnitud.
fig, ax = plt.subplots(figsize=(9.2, 4.6), constrained_layout=True)
for label, (b, a) in {
    "Pasa bajos Butterworth": butter(5, 65, btype="lowpass", fs=FS),
    "Pasa altos Chebyshev II": cheby2(5, 35, 65, btype="highpass", fs=FS),
    "Pasa bandas FIR Kaiser": (taps_band, [1.0]),
}.items():
    freq, h = freqz(b, a, worN=8192, fs=FS)
    ax.plot(freq, 20 * np.log10(np.maximum(np.abs(h), 1e-7)), label=label, color=colors[label], lw=1.5)
ax.axvline(65, color="#444444", ls=":", lw=1)
ax.set(xlim=(0, 300), ylim=(-100, 5), xlabel="Frecuencia (Hz)", ylabel="Magnitud (dB)", title="Respuesta en frecuencia de los filtros")
ax.legend(frameon=False, loc="lower left")
fig.savefig(PLOTS / "respuesta_filtros.png", bbox_inches="tight")
plt.close(fig)

# 3. Ventana temporal: una zona con ciclos e impactos amortiguados.
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
window = (t >= 2.0) & (t <= 2.65)
axes[0].plot(t[window], observed[window], color="#29434e", lw=0.7)
axes[0].set_ylabel("Entrada")
for ax, (label, output) in zip(axes[1:], filtered.items()):
    ax.plot(t[window], output[window], color=colors[label], lw=0.8)
    ax.plot(t[window], clean[window], color="#777777", lw=0.6, ls="--", label="Referencia")
    ax.set_ylabel(label.replace(" ", "\n", 1))
    ax.legend(frameon=False, loc="upper right")
axes[-1].set_xlabel("Tiempo (s)")
axes[0].set_title("Comparación temporal durante la vibración")
fig.savefig(PLOTS / "comparacion_temporal.png", bbox_inches="tight")
plt.close(fig)

# 4. Espectros de las salidas.
fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
f, p = welch(observed[mask], fs=FS, nperseg=4096)
ax.plot(f, 10 * np.log10(p + 1e-14), color="#888888", lw=1, label="Entrada")
for label, output in filtered.items():
    f, p = welch(output[mask], fs=FS, nperseg=4096)
    ax.plot(f, 10 * np.log10(p + 1e-14), lw=1.2, color=colors[label], label=label)
ax.set(xlim=(0, 300), xlabel="Frecuencia (Hz)", ylabel="PSD (dB/Hz)", title="Espectros antes y después del filtrado")
ax.legend(frameon=False, ncol=2)
fig.savefig(PLOTS / "espectros_salida.png", bbox_inches="tight")
plt.close(fig)

print(metrics.to_string(index=False))
print(f"SNR de entrada: {summary['SNR_entrada_dB']:.3f} dB")
