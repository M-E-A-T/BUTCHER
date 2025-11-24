#!/usr/bin/env python3
import sounddevice as sd
import numpy as np
import sys

# --------------------------
# CONFIG
# --------------------------

HOP_S = 1024              # block size / FFT size (power of 2 is nice)

# Make silence gate VERY forgiving so we don't kill real signals
SILENCE_RMS = 1e-4       # was 1e-4

F_MIN = 20.0              # ignore DC / subsonic
F_MAX = 20000.0           # upper limit for search

FREQ_SMOOTH_ALPHA = 0.2   # 0 = no smoothing, closer to 1 = slower, smoother
smoothed_freq = None

selected_device = None
selected_channel = 0
num_channels = None
SAMPLE_RATE = None        # will be set from chosen device

# Toggle this to see levels
DEBUG_LEVEL = True


# --------------------------
# DEVICE SELECTION
# --------------------------

def list_audio_devices():
    print("\n" + "="*60)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("="*60)

    devices = sd.query_devices()
    input_devices = []

    for idx, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append(idx)
            print(f"[{idx}] {device['name']}")
            print(f"    Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {int(device['default_samplerate'])} Hz")
            print()

    return input_devices


def select_device():
    global num_channels, SAMPLE_RATE

    input_devices = list_audio_devices()
    if not input_devices:
        print("ERROR: No input devices found!")
        sys.exit(1)

    # choose device
    while True:
        try:
            choice = int(input("Select device index: "))
            if choice in input_devices:
                device_info = sd.query_devices(choice)
                print(f"\nSelected: {device_info['name']}")
                sr = int(device_info['default_samplerate'])
                SAMPLE_RATE = sr
                print(f"Default Sample Rate: {sr} Hz")
                print(f"Available Channels: {device_info['max_input_channels']}")
                device_idx = choice
                break
            else:
                print(f"Invalid index. Choose from: {input_devices}")
        except ValueError:
            print("Enter a number.")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)

    max_channels = device_info['max_input_channels']
    num_channels = max_channels

    # choose channel
    if max_channels > 1:
        print(f"\nDevice has {max_channels} input channels")
        for i in range(max_channels):
            print(f"  [{i+1}] Input {i+1} (Channel {i+1})")

        while True:
            try:
                ch = int(input(f"Select input (1-{max_channels}): "))
                if 1 <= ch <= max_channels:
                    channel = ch - 1
                    print(f"Using Input {ch}\n")
                    return device_idx, channel
                else:
                    print(f"Enter a number between 1 and {max_channels}")
            except ValueError:
                print("Enter a number.")
    else:
        print("Using single available input (Channel 1)\n")
        return device_idx, 0


# --------------------------
# DOMINANT FREQUENCY
# --------------------------

def dominant_frequency(mono_block: np.ndarray, sample_rate: int) -> float:
    """
    Compute dominant frequency (Hz) in mono_block using magnitude spectrum.
    Returns 0.0 if no valid peak is found.
    """
    x = mono_block.astype(np.float32)

    # Hann window to reduce spectral leakage
    window = np.hanning(len(x))
    x_win = x * window

    # Real FFT
    spectrum = np.fft.rfft(x_win)
    mags = np.abs(spectrum)

    # Frequency axis
    freqs = np.fft.rfftfreq(len(x_win), 1.0 / sample_rate)

    # Limit search to [F_MIN, F_MAX]
    mask = (freqs >= F_MIN) & (freqs <= F_MAX)
    if not np.any(mask):
        return 0.0

    mags_roi = mags[mask]
    freqs_roi = freqs[mask]

    if mags_roi.size == 0:
        return 0.0

    idx_peak = np.argmax(mags_roi)
    peak_freq = float(freqs_roi[idx_peak])
    return peak_freq


# --------------------------
# AUDIO CALLBACK
# --------------------------

def audio_callback(indata, frames, time_info, status):
    global smoothed_freq

    if status:
        print(status, file=sys.stderr)

    # mono from selected channel
    if indata.shape[1] > 1:
        mono = indata[:, selected_channel].astype(np.float32)
    else:
        mono = indata[:, 0].astype(np.float32)

    # silence check
    rms = np.sqrt(np.mean(mono**2) + 1e-12)

    if DEBUG_LEVEL:
        # show RMS so you can see if audio is actually arriving
        # comment out later if spammy
        print(f"RMS: {rms:.8f}", file=sys.stderr)

    if rms < SILENCE_RMS:
        print("0.0", flush=True)
        return

    # compute dominant frequency for this block
    freq_block = dominant_frequency(mono, SAMPLE_RATE)

    # if something went wrong, just print 0
    if freq_block <= 0.0:
        print("0.0", flush=True)
        return

    # smoothing
    if smoothed_freq is None:
        smoothed_freq = freq_block
    else:
        a = FREQ_SMOOTH_ALPHA
        smoothed_freq = (1.0 - a) * smoothed_freq + a * freq_block

    print(f"{smoothed_freq:.2f}", flush=True)


# --------------------------
# MAIN
# --------------------------

def main():
    global selected_device, selected_channel, num_channels

    print("\n" + "="*60)
    print("Real-time Dominant Frequency Detector")
    print("="*60)
    print("Prints dominant frequency in Hz (0 when silent).")
    print("Press Ctrl+C to stop.\n")

    selected_device, selected_channel = select_device()

    print("Starting audio stream...")
    print(f"Device index: {selected_device}, Channel: {selected_channel+1}, SR: {SAMPLE_RATE}\n")

    try:
        with sd.InputStream(
            device=selected_device,
            channels=num_channels,
            samplerate=SAMPLE_RATE,
            blocksize=HOP_S,
            callback=audio_callback,
            dtype='float32'
        ):
            print("Listening and estimating dominant frequency...\n")
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print("Error:", e)
    finally:
        print("Audio stream closed.")


if __name__ == "__main__":
    main()
