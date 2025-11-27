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
OSC_ADDR      = "/flux"          # OSC address for FLUX
OSC_LOCAL_IP  = "127.0.0.1"      # same machine
OSC_BCAST_IP  = "7.7.7.255"  # LAN broadcast

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

stop_flag = False   # <-- press q to set this and stop


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

    # --------- SELECT DEVICE INDEX ----------
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
        except ValueError:
            print("Enter a valid number.")
        except EOFError:
            print("\nEOF on stdin, exiting.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    max_channels = chans
    num_channels = max_channels
    audioInputSampleRate = sr
    selected_channel = None

    # --------- SELECT CHANNEL ----------
    if max_channels > 1:
        print(f"\nDevice has {max_channels} input channels\n")

        for i in range(max_channels):
            print(f"  [{i+1}] Input {i+1}")

        while True:
            try:
                channel_choice = input(f"\nSelect input (1-{max_channels}) or 'q' to quit: ").strip()
                if channel_choice.lower() == 'q':
                    print("Exiting by user request.")
                    sys.exit(0)

                channel_num = int(channel_choice)

                if 1 <= channel_num <= max_channels:
                    selected_channel = channel_num - 1
                    print(f"Using Input {channel_num}\n")
                    break
                else:
                    print(f"Enter a number between 1 and {max_channels}")
            except ValueError:
                print("Enter a valid number.")
            except EOFError:
                print("\nEOF on stdin, exiting.")
                sys.exit(1)
            except KeyboardInterrupt:
                print("\nExiting...")
                sys.exit(0)
    else:
        selected_channel = 0
        print("Using single input channel\n")

    return device_idx


# ==============================
# FLUX SYSTEM
# ==============================

def compute_spectral_flux(signal):
    global prev_spectrum

    # Hanning window
    window = np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal * window))

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

    if stop_flag:
        # Tell PyAudio to stop the stream
        return (in_data, pyaudio.paComplete)

    # Convert bytes -> float32 array
    signal = np.frombuffer(in_data, dtype=np.float32)

    # Handle multi-channel: de-interleave & pick the selected channel
    if num_channels and num_channels > 1:
        try:
            signal = signal.reshape(-1, num_channels)[:, selected_channel]
        except ValueError:
            # fallback if reshape fails
            signal = signal[::num_channels]

    # Compute spectral flux
    flux_raw = compute_spectral_flux(signal)

    # Smooth it
    if smoothed_flux is None:
        smoothed_flux = flux_raw
    else:
        smoothed_flux = (1 - FLUX_SMOOTH) * smoothed_flux + FLUX_SMOOTH * flux_raw

    flux_int = int(smoothed_flux)

    # Print
    print(f"{flux_int}", flush=True)

    # OSC send
    osc_local.send_message(OSC_ADDR, flux_int)
    osc_bcast.send_message(OSC_ADDR, flux_int)

    return (in_data, pyaudio.paContinue)


# ==============================
# KEYBOARD LISTENER (q to quit)
# ==============================

def keyboard_listener():
    """
    Waits for 'q' + Enter on stdin and sets stop_flag.
    This lets you quit cleanly without Ctrl+C.
    """
    global stop_flag
    try:
        while not stop_flag:
            line = sys.stdin.readline()
            if not line:
                # EOF (e.g. non-interactive env) -> just exit listener
                break
            if line.strip().lower() == 'q':
                print("\n[q] Quit requested.")
                stop_flag = True
                break
    except Exception:
        # Don't kill the main program if stdin is weird
        pass


# ==============================
# MAIN
# ==============================

def main():
    global selected_device_index, stop_flag

    print("\nReal-time Spectral Flux → OSC (/flux)")
    print("Prints Flux every block")
    print(f"Sending OSC on {OSC_ADDR}")
    print(f"  → local:     {OSC_LOCAL_IP}:{OSC_PORT}")
    print(f"  → broadcast: {OSC_BCAST_IP}:{OSC_PORT}")
    print("Press 'q' then Enter to stop (after the stream starts), or Ctrl+C anytime.\n")

    pa = pyaudio.PyAudio()

    try:
        # 1) First, interactively choose device & channel
        selected_device_index = select_device(pa)

        # 2) AFTER THAT, start the keyboard listener
        kb_thread = threading.Thread(target=keyboard_listener, daemon=True)
        kb_thread.start()

        # 3) Create and start the input stream
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

        # Main loop: run until stream stops or stop_flag is set
        while inputStream.is_active() and not stop_flag:
            sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Ctrl+C] Stopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        stop_flag = True
        try:
            if 'inputStream' in locals():
                inputStream.stop_stream()
                inputStream.close()
        except Exception:
            pass

        pa.terminate()
        print("Audio stream closed.")
        print("OSC clients cleaned up (no server port was bound).")


if __name__ == "__main__":
    main()
