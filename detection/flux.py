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
OSC_MODE_ADDR = "/butcher/mode"    # NEW MODE ENDPOINT

OSC_LOCAL_IP  = "127.0.0.1"
OSC_BCAST_IP  = "7.7.7.255"        # LAN broadcast

osc_local = SimpleUDPClient(OSC_LOCAL_IP, OSC_PORT)
osc_bcast = SimpleUDPClient(OSC_BCAST_IP, OSC_PORT, allow_broadcast=True)

# Globals
selected_device_index = None
audioInputSampleRate  = None
selected_channel      = None
num_channels          = None

prev_spectrum  = None
smoothed_flux  = None
FLUX_SMOOTH    = 0.2

stop_flag = False

# dynamic normalization globals
flux_min = 1e9
flux_max = 0.0


# ==============================
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

def compute_spectral_flux(signal):
    global prev_spectrum

    # Hanning window
    window = np.hanning(len(signal))
    x = signal * window

    # Normalize by RMS → gain invariant
    rms = np.sqrt(np.mean(x**2))
    if rms > 1e-8:
        x = x / rms

    spectrum = np.abs(np.fft.rfft(x))

    if prev_spectrum is None:
        prev_spectrum = spectrum
        return 0.0

    diff = spectrum - prev_spectrum
    diff[diff < 0] = 0

    flux_value = np.sum(diff)
    prev_spectrum = spectrum

    return float(flux_value)


# ==============================
# AUDIO CALLBACK
# ==============================

def readAudioFrames(in_data, frame_count, time_info, status):
    global smoothed_flux, selected_channel, num_channels, stop_flag
    global flux_min, flux_max

    if stop_flag:
        return (in_data, pyaudio.paComplete)

    # Convert bytes → float32
    signal = np.frombuffer(in_data, dtype=np.float32)

    # Handle multi-channel
    if num_channels and num_channels > 1:
        try:
            signal = signal.reshape(-1, num_channels)[:, selected_channel]
        except:
            signal = signal[::num_channels]

    # Compute flux
    flux_raw = compute_spectral_flux(signal)

    # Smooth it
    if smoothed_flux is None:
        smoothed_flux = flux_raw
    else:
        smoothed_flux = (1 - FLUX_SMOOTH) * smoothed_flux + FLUX_SMOOTH * flux_raw

    # ==============================
    # DYNAMIC NORMALIZATION: 0..1000
    # ==============================
    if smoothed_flux > 1e-6:
        flux_min = min(flux_min, smoothed_flux)
        flux_max = max(flux_max, smoothed_flux)

    if flux_max > flux_min:
        norm = (smoothed_flux - flux_min) / (flux_max - flux_min)
    else:
        norm = 0.0

    norm = np.clip(norm, 0.0, 1.0)
    flux_int = int(norm * 1000)

    print(f"{flux_int}", flush=True)

    # Send OSC
    osc_local.send_message(OSC_ADDR, flux_int)
    osc_bcast.send_message(OSC_ADDR, flux_int)

    return (in_data, pyaudio.paContinue)


# ==============================
# MODE SENDER (cycle every 30 sec)
# ==============================

def mode_sender():
    """
    Cycle through 1 → 2 → 3 → 1 ... every 30 seconds
    and send to /butcher/mode on both local + broadcast.
    """
    mode_val = 1
    while not stop_flag:
        print(f"[MODE] Sending {mode_val} on {OSC_MODE_ADDR}", flush=True)
        try:
            osc_local.send_message(OSC_MODE_ADDR, mode_val)
            osc_bcast.send_message(OSC_MODE_ADDR, mode_val)
        except Exception as e:
            print(f"[MODE] Error sending mode: {e}")

        sleep(30.0)

        # cycle to next mode
        mode_val = (mode_val % 3) + 1


# ==============================
# KEYBOARD LISTENER
# ==============================

def keyboard_listener():
    global stop_flag
    try:
        while not stop_flag:
            line = sys.stdin.readline()
            if not line:
                break
            if line.strip().lower() == 'q':
                print("\n[q] Quit requested.")
                stop_flag = True
                break
    except:
        pass


# ==============================
# MAIN
# ==============================

def main():
    global selected_device_index, stop_flag

    print("\nReal-time Spectral Flux → OSC (/butcher/flux)")
    print("Mode cycling → OSC (/butcher/mode) every 30 seconds")
    print("Sending OSC:")
    print(f"  → local:     {OSC_LOCAL_IP}:{OSC_PORT}")
    print(f"  → broadcast: {OSC_BCAST_IP}:{OSC_PORT}\n")

    pa = pyaudio.PyAudio()

    try:
        selected_device_index = select_device(pa)

        kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
        kb_thread.start()

        # Start mode cycling thread
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
