#!/usr/bin/env python3

import pyaudio
import numpy as np
import sys
from collections import deque
from time import sleep
from pythonosc.udp_client import SimpleUDPClient
import threading

# ==============================
# CONFIG
# ==============================

BUFFER_SIZE = 512
WINDOW_SIZE_MULTIPLE = 1   # not used for flux, but left for consistency

# OSC CONFIG
OSC_PORT      = 9000

OSC_ADDR      = "/butcher/flux"
OSC_ADDR_LOW  = "/butcher/flux_low"
OSC_ADDR_MID  = "/butcher/flux_mid"
OSC_ADDR_HIGH = "/butcher/flux_high"
OSC_MODE_ADDR = "/butcher/mode"    # MODE ENDPOINT

OSC_LOCAL_IP  = "127.0.0.1"
OSC_BCAST_IP  = "7.7.7.255"        # LAN broadcast

osc_local = SimpleUDPClient(OSC_LOCAL_IP, OSC_PORT)
osc_bcast = SimpleUDPClient(OSC_BCAST_IP, OSC_PORT, allow_broadcast=True)

# Globals
selected_device_index = None
audioInputSampleRate  = None
selected_channel      = None
num_channels          = None

prev_spectrum      = None
smoothed_flux      = None
FLUX_SMOOTH        = 0.2


prev_spectrum_bands = [None, None, None]
# For new normalization: rolling max for each band
peak_bands = [1e-6, 1e-6, 1e-6]
# Faster decay for low/mid bands to better catch transients
PEAK_DECAY_BANDS = [0.98, 0.98, 0.995]  # [low, mid, high]

# Thresholds for low/mid bands to suppress noise
LOW_BAND_THRESHOLD = 1e-5
MID_BAND_THRESHOLD = 1e-5

# === TRANSIENT SENSITIVITY (edit these to adjust per-band sensitivity) ===
# Lower = more sensitive, Higher = less sensitive
TRANSIENT_THRESHOLDS = [900, 950, 500]  # [low, mid, high]

stop_flag = False


# ==============================40
# DEVICE SELECTION
# ==============================

def list_audio_devices(pa):
    print("\n" + "="*60)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("="*60)

    input_devices = []
    device_count = pa.get_device_count()

    for idx in range(device_count):
        info = pa.get_device_info_by_index(idx)
        if info.get('maxInputChannels', 0) > 0:
            input_devices.append(idx)
            name = info.get('name', 'Unknown')
            chans = info.get('maxInputChannels', 0)
            sr = int(info.get('defaultSampleRate', 0))

            print(f"[{idx}] {name}")
            print(f"    Channels: {chans}, Sample Rate: {sr} Hz\n")

    return input_devices


def select_device(pa):
    global num_channels, selected_channel, audioInputSampleRate

    input_devices = list_audio_devices(pa)

    if not input_devices:
        print("ERROR: No input devices found!")
        sys.exit(1)

    # --------- SELECT DEVICE ----------
    while True:
        try:
            choice = input("Select device index (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print("Exiting by user request.")
                sys.exit(0)

            device_idx = int(choice)

            if device_idx in input_devices:
                device_info = pa.get_device_info_by_index(device_idx)
                name = device_info.get('name', 'Unknown')
                sr = int(device_info.get('defaultSampleRate', 0))
                chans = device_info.get('maxInputChannels', 0)

                print(f"\nSelected: {name}")
                print(f"Sample Rate: {sr} Hz")
                print(f"Available Channels: {chans}")
                break

            else:
                print(f"Invalid device index. Choose from: {input_devices}")

        except:
            print("Enter a valid number.")

    num_channels = chans
    audioInputSampleRate = sr

    # --------- SELECT CHANNEL ----------
    if num_channels > 1:
        print(f"\nDevice has {num_channels} input channels\n")
        for i in range(num_channels):
            print(f"  [{i+1}] Input {i+1}")

        while True:
            try:
                channel_choice = input(f"\nSelect input (1-{num_channels}) or 'q' to quit: ").strip()
                if channel_choice.lower() == 'q':
                    print("Exiting by user request.")
                    sys.exit(0)

                channel_num = int(channel_choice)
                if 1 <= channel_num <= num_channels:
                    selected_channel = channel_num - 1
                    print(f"Using Input {channel_num}\n")
                    break
                else:
                    print(f"Enter a number between 1 and {num_channels}")

            except:
                print("Enter a valid number.")
    else:
        selected_channel = 0
        print("Using single input channel\n")

    return device_idx


# ==============================
# FLUX SYSTEM (RMS-normalized)
# ==============================

def compute_spectral_flux(signal, prev_spectrum):
    # Hanning window
    window = np.hanning(len(signal))
    x = signal * window

    spectrum = np.abs(np.fft.rfft(x))

    if prev_spectrum is None:
        prev_spectrum = spectrum
        return 0.0, prev_spectrum

    diff = spectrum - prev_spectrum
    diff[diff < 0] = 0

    flux_value = np.sum(diff)
    prev_spectrum = spectrum

    return float(flux_value), prev_spectrum

def split_bands(signal, sample_rate):
    # FFT-based band split
    spectrum = np.fft.rfft(signal * np.hanning(len(signal)))
    freqs = np.fft.rfftfreq(len(signal), 1/sample_rate)
    # Narrow low band for kick: 40-120 Hz
    # Narrow mid band for snare: 800-2000 Hz
    bands = [(40, 120), (800, 2200), (10000, sample_rate//2)] # Lows, Mids, Highs
    band_signals = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_spectrum = np.zeros_like(spectrum)
        band_spectrum[mask] = spectrum[mask]
        band_signal = np.fft.irfft(band_spectrum, n=len(signal))
        band_signals.append(band_signal)
    return band_signals


# ==============================
# AUDIO CALLBACK
# ==============================

def readAudioFrames(in_data, frame_count, time_info, status):
    global smoothed_flux, selected_channel, num_channels, stop_flag
    global prev_spectrum_bands, peak_bands

    if stop_flag:
        return (in_data, pyaudio.paComplete)

    # Convert bytes → float32
    signal = np.frombuffer(in_data, dtype=np.float32)

    # Sum to mono if multi-channel
    if num_channels and num_channels > 1:
        try:
            signal = signal.reshape(-1, num_channels)
            signal = np.mean(signal, axis=1)
        except:
            signal = signal[::num_channels]

    # Split into 3 bands
    band_signals = split_bands(signal, audioInputSampleRate)

    # Transient gate: output a fixed value for a short duration when a transient is detected
    TRANSIENT_HOLD_FRAMES = 6  # Number of callback frames to hold the output (e.g., ~70ms at 512/44.1kHz)
    TRANSIENT_OUTPUT_VALUE = 500
    if not hasattr(readAudioFrames, 'transient_counters'):
        readAudioFrames.transient_counters = [0, 0, 0]
    flux_ints = []
    for i, band_signal in enumerate(band_signals):
        flux_raw, prev_spectrum_bands[i] = compute_spectral_flux(band_signal, prev_spectrum_bands[i])
        # Threshold for low/mid bands to suppress noise
        if i == 0 and flux_raw < LOW_BAND_THRESHOLD:
            flux_raw = 0.0
        if i == 1 and flux_raw < MID_BAND_THRESHOLD:
            flux_raw = 0.0
        # New normalization: rolling peak hold (decaying max)
        if flux_raw > peak_bands[i]:
            peak_bands[i] = flux_raw
        else:
            peak_bands[i] *= PEAK_DECAY_BANDS[i]
        # Normalize to 0-1000, preserve transients
        norm = flux_raw / (peak_bands[i] + 1e-8)
        norm = np.clip(norm, 0.0, 1.0)
        flux_int = int(norm * 1000)
        # Transient detection and hold logic
        if flux_int >= TRANSIENT_THRESHOLDS[i]:
            readAudioFrames.transient_counters[i] = TRANSIENT_HOLD_FRAMES
        elif readAudioFrames.transient_counters[i] > 0:
            readAudioFrames.transient_counters[i] -= 1
        if readAudioFrames.transient_counters[i] > 0:
            flux_ints.append(TRANSIENT_OUTPUT_VALUE)
        else:
            flux_ints.append(0)

    print(f"Low:{flux_ints[0]} Mid:{flux_ints[1]} High:{flux_ints[2]}", flush=True)

    # Send OSC for each band
    osc_local.send_message(OSC_ADDR_LOW, flux_ints[0])
    osc_local.send_message(OSC_ADDR_MID, flux_ints[1])
    osc_local.send_message(OSC_ADDR_HIGH, flux_ints[2])
    osc_bcast.send_message(OSC_ADDR_LOW, flux_ints[0])
    osc_bcast.send_message(OSC_ADDR_MID, flux_ints[1])
    osc_bcast.send_message(OSC_ADDR_HIGH, flux_ints[2])

    return (in_data, pyaudio.paContinue)


# ==============================
# MODE SENDER (auto cycle)
# ==============================

def send_mode_value(mode_val):
    """Send a mode value (1/2/3) to /butcher/mode on both local + broadcast."""
    try:
        print(f"[MODE] Sending {mode_val} on {OSC_MODE_ADDR}", flush=True)
        osc_local.send_message(OSC_MODE_ADDR, mode_val)
        osc_bcast.send_message(OSC_MODE_ADDR, mode_val)
    except Exception as e:
        print(f"[MODE] Error sending mode {mode_val}: {e}")


def mode_sender():
    """
    Cycle through 1 → 2 → 3 → 1 ... every 30 seconds
    and send to /butcher/mode on both local + broadcast.
    """
    mode_val = 1
    while not stop_flag:
        send_mode_value(mode_val)
        sleep(30.0)
        mode_val = (mode_val % 3) + 1


# ==============================
# KEYBOARD LISTENER
# ==============================

def keyboard_listener():
    """
    Read lines from stdin:

      '1', '2', '3' → send /butcher/mode with that value (manual override)
      'q'           → quit the whole script
    """
    global stop_flag
    try:
        print("\nKeyboard controls:")
        print("  1 / 2 / 3  → send mode to /butcher/mode")
        print("  q          → quit\n")

        while not stop_flag:
            line = sys.stdin.readline()
            if not line:
                break

            cmd = line.strip().lower()
            if cmd == 'q':
                print("\n[q] Quit requested.")
                stop_flag = True
                break
            elif cmd in ('1', '2', '3'):
                mode_val = int(cmd)
                send_mode_value(mode_val)
            # ignore other inputs
    except Exception as e:
        print(f"[KEYBOARD] Error: {e}")


# ==============================
# MAIN
# ==============================

def main():
    global selected_device_index, stop_flag

    print("\nReal-time Spectral Flux → OSC (/butcher/flux)")
    print("Mode cycling → OSC (/butcher/mode) every 30 seconds")
    print("Manual modes: type 1 / 2 / 3 + Enter to override\n")
    print("Sending OSC:")
    print(f"  → local:     {OSC_LOCAL_IP}:{OSC_PORT}")
    print(f"  → broadcast: {OSC_BCAST_IP}:{OSC_PORT}\n")

    pa = pyaudio.PyAudio()

    try:
        selected_device_index = select_device(pa)

        # Keyboard thread (manual modes + quit)
        kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
        kb_thread.start()

        # Auto mode cycling thread
        mode_thread = threading.Thread(target=mode_sender, daemon=True)
        mode_thread.start()

        inputStream = pa.open(
            format=pyaudio.paFloat32,
            input=True,
            channels=num_channels,
            input_device_index=selected_device_index,
            frames_per_buffer=BUFFER_SIZE,
            rate=audioInputSampleRate,
            stream_callback=readAudioFrames
        )

        inputStream.start_stream()

        while inputStream.is_active() and not stop_flag:
            sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        stop_flag = True
        try:
            inputStream.stop_stream()
            inputStream.close()
        except:
            pass

        pa.terminate()
        print("Audio stream closed.")
        print("Done.")


if __name__ == "__main__":
    main()
