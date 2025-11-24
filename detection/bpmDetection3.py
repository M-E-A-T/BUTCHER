#!/usr/bin/env python3

import pyaudio
import aubio
import numpy as np
import sys
from collections import deque
from time import sleep
from pythonosc.udp_client import SimpleUDPClient

# ==============================
# CONFIG
# ==============================

BUFFER_SIZE = 512
WINDOW_SIZE_MULTIPLE = 4   # 2 or 4 for higher accuracy (more CPU)

SILENCE_RMS    = 0.001     # tweak depending on your input level
MIN_CONFIDENCE = 0.1       # aubio confidence threshold
SMOOTH_HISTORY = 8         # running average length for BPM smoothing

# OSC CONFIG (local + broadcast)
OSC_PORT      = 9000
OSC_ADDR      = "/bpm"          # OSC address for BPM
OSC_LOCAL_IP  = "127.0.0.1"     # same machine
OSC_BCAST_IP  = "192.168.8.255" # everyone on your 192.168.8.x LAN

osc_local = SimpleUDPClient(OSC_LOCAL_IP, OSC_PORT)
osc_bcast = SimpleUDPClient(OSC_BCAST_IP, OSC_PORT, allow_broadcast=True)

# Globals
selected_device_index = None
audioInputSampleRate  = None
selected_channel      = None
num_channels          = None

tempoDetection = None
bpm_history    = deque(maxlen=SMOOTH_HISTORY)


# ==============================
# DEVICE SELECTION (INPUT LOGIC)
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

    # Select device index
    while True:
        try:
            choice = input("Select device index: ")
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
        except ValueError:
            print("Enter a valid number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    max_channels = chans
    num_channels = max_channels
    audioInputSampleRate = sr
    selected_channel = None

    # Choose channel
    if max_channels > 1:
        print(f"\nDevice has {max_channels} input channels\n")

        for i in range(max_channels):
            print(f"  [{i+1}] Input {i+1}")

        while True:
            try:
                channel_choice = input(f"\nSelect input (1-{max_channels}): ")
                channel_num = int(channel_choice)

                if 1 <= channel_num <= max_channels:
                    selected_channel = channel_num - 1
                    print(f"Using Input {channel_num}\n")
                    break
                else:
                    print(f"Enter a number between 1 and {max_channels}")
            except ValueError:
                print("Enter a valid number.")
            except KeyboardInterrupt:
                print("\nExiting")
                sys.exit(0)
    else:
        selected_channel = 0
        print("Using single input channel\n")

    return device_idx


# ==============================
# BPM SYSTEM
# ==============================

def init_tempo_detection():
    global tempoDetection

    hopSize = BUFFER_SIZE
    winSize = hopSize * WINDOW_SIZE_MULTIPLE

    tempoDetection = aubio.tempo(
        method='default',
        buf_size=winSize,
        hop_size=hopSize,
        samplerate=audioInputSampleRate
    )


def smooth_bpm(bpm):
    bpm_history.append(bpm)
    return sum(bpm_history) / len(bpm_history)


# ==============================
# AUDIO CALLBACK
# ==============================

def readAudioFrames(in_data, frame_count, time_info, status):
    global tempoDetection, selected_channel, num_channels

    # Convert bytes -> float32 array
    signal = np.frombuffer(in_data, dtype=np.float32)

    # Handle multi-channel: de-interleave & pick the selected channel
    if num_channels and num_channels > 1:
        try:
            signal = signal.reshape(-1, num_channels)[:, selected_channel]
        except ValueError:
            # fallback if reshape fails
            signal = signal[::num_channels]
    else:
        # already mono
        pass

    # Silence check
    rms = np.sqrt(np.mean(signal**2) + 1e-12)

    # Run aubio tempo detection
    _ = tempoDetection(signal)
    bpm = float(tempoDetection.get_bpm())
    conf = float(tempoDetection.get_confidence())

    # Decide what to emit (every callback)
    if rms < SILENCE_RMS or conf < MIN_CONFIDENCE or bpm <= 0:
        # No reliable BPM: send 0
        print("0", flush=True)
        osc_local.send_message(OSC_ADDR, 0.0)
        osc_bcast.send_message(OSC_ADDR, 0.0)
    else:
        bpm_out = smooth_bpm(bpm)

        # --------- DOUBLE UNDER 100 BPM ---------
        if bpm_out > 0 and bpm_out < 100:
            bpm_out *= 2.0
        # ---------------------------------------

        bpm_out = int(bpm_out)
        print(f"{bpm_out:.2f}", flush=True)

        # Send just the BPM number via OSC
        osc_local.send_message(OSC_ADDR, bpm_out)
        osc_bcast.send_message(OSC_ADDR, bpm_out)

    return (in_data, pyaudio.paContinue)


# ==============================
# MAIN
# ==============================

def main():
    global selected_device_index

    print("\nReal-time BPM Detector → OSC (/bpm)")
    print("Prints BPM every block, 0 when no music")
    print("Any BPM under 100 gets doubled before output.")
    print(f"Sending OSC on {OSC_ADDR}")
    print(f"  → local:     {OSC_LOCAL_IP}:{OSC_PORT}")
    print(f"  → broadcast: {OSC_BCAST_IP}:{OSC_PORT}")
    print("Press Ctrl+C to stop.\n")

    pa = pyaudio.PyAudio()

    try:
        selected_device_index = select_device(pa)
        init_tempo_detection()

        # Create and start the input stream
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

        # Run until Ctrl+C
        while inputStream.is_active():
            sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        try:
            if 'inputStream' in locals():
                inputStream.stop_stream()
                inputStream.close()
        except Exception:
            pass
        pa.terminate()
        print("Audio stream closed.")


if __name__ == "__main__":
    main()
