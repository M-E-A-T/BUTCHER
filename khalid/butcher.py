#!/usr/bin/env python3
import mido
import sounddevice as sd
import numpy as np
import sys
import threading

BUFFER_SIZE = 2048
selected_device = None
sample_rate = None
selected_channel = None
mode = None

kick_threshold = 100.0
snare_threshold = 100.0
hihat_threshold = 100.0
kick_debounce = 0.12        # 120ms
snare_debounce = 0.08       # 80ms
hihat_debounce = 0.05       # 50ms
min_energy = 10.0

outports = mido.get_output_names()
out = None


last_detection = {'kick': 0, 'snare': 0, 'hihat': 0}
start_time = None

def select_mode():
    print("\n" + "="*60)
    print("SELECT MODE")
    print("="*60)
    print("[1] Debug")
    print("[2] Butcher")
    print()
    
    while True:
        try:
            choice = input("Select mode (1-2): ")
            
            if choice == "1":
                print("\nDebug selected\n")
                return "debug"
            elif choice == "2":
                print("\nButcher selected\n")
                return "butcher"
            else:
                print("enter 1 or 2")
        except KeyboardInterrupt:
            print("\nExiting")
            sys.exit(0)


def show_help():
    print("\n" + "="*60)
    print("LIVE PARAMETER CONTROLS")
    print("="*60)
    print("SENSITIVITY (energy thresholds):")
    print("  k+ / k-  : Increase/decrease kick threshold")
    print("  s+ / s-  : Increase/decrease snare threshold")
    print("  h+ / h-  : Increase/decrease hi-hat threshold")
    print()
    print("DEBOUNCE (minimum time between triggers):")
    print("  K+ / K-  : Increase/decrease kick debounce time")
    print("  S+ / S-  : Increase/decrease snare debounce time")
    print("  H+ / H-  : Increase/decrease hi-hat debounce time")
    print()
    print("OTHER:")
    print("  m+ / m-  : Increase/decrease minimum energy threshold")
    print("  midi     : Open MIDI out port")
    print("  show     : Display current settings")
    print("  help     : Show this menu")
    print("="*60 + "\n")


def show_settings():
    print("\n" + "="*60)
    print("CURRENT SETTINGS")
    print("="*60)
    print(f"Kick Threshold:    {kick_threshold:.1f}  |  Debounce: {kick_debounce*1000:.0f}ms")
    print(f"Snare Threshold:   {snare_threshold:.1f}  |  Debounce: {snare_debounce*1000:.0f}ms")
    print(f"Hi-hat Threshold:  {hihat_threshold:.1f}  |  Debounce: {hihat_debounce*1000:.0f}ms")
    print(f"Minimum Energy:    {min_energy:.1f}")
    print("="*60 + "\n")


def parameter_control_thread():
    global kick_threshold, snare_threshold, hihat_threshold
    global kick_debounce, snare_debounce, hihat_debounce, min_energy
    global out, outports 
    
    print("\ntype 'help' for parameter controls, 'show' for current settings\n")
    
    while True:
        try:
            cmd = input().strip()
            
            if cmd == "k+":
                kick_threshold += 10.0
                print(f"Kick threshold: {kick_threshold:.1f}")
            elif cmd == "k-":
                kick_threshold = max(10.0, kick_threshold - 10.0)
                print(f"Kick threshold: {kick_threshold:.1f}")
            elif cmd == "s+":
                snare_threshold += 10.0
                print(f"Snare threshold: {snare_threshold:.1f}")
            elif cmd == "s-":
                snare_threshold = max(10.0, snare_threshold - 10.0)
                print(f"Snare threshold: {snare_threshold:.1f}")
            elif cmd == "h+":
                hihat_threshold += 10.0
                print(f"Hi-hat threshold: {hihat_threshold:.1f}")
            elif cmd == "h-":
                hihat_threshold = max(10.0, hihat_threshold - 10.0)
                print(f"Hi-hat threshold: {hihat_threshold:.1f}")
            
            elif cmd == "K+":
                kick_debounce = min(0.30, kick_debounce + 0.01)
                print(f"Kick debounce: {kick_debounce*1000:.0f}ms")
            elif cmd == "K-":
                kick_debounce = max(0.02, kick_debounce - 0.01)
                print(f"Kick debounce: {kick_debounce*1000:.0f}ms")
            elif cmd == "S+":
                snare_debounce = min(0.30, snare_debounce + 0.01)
                print(f"Snare debounce: {snare_debounce*1000:.0f}ms")
            elif cmd == "S-":
                snare_debounce = max(0.02, snare_debounce - 0.01)
                print(f"Snare debounce: {snare_debounce*1000:.0f}ms")
            elif cmd == "H+":
                hihat_debounce = min(0.30, hihat_debounce + 0.01)
                print(f"Hi-hat debounce: {hihat_debounce*1000:.0f}ms")
            elif cmd == "H-":
                hihat_debounce = max(0.02, hihat_debounce - 0.01)
                print(f"Hi-hat debounce: {hihat_debounce*1000:.0f}ms")
            
            elif cmd == "m+":
                min_energy += 5.0
                print(f"Minimum energy: {min_energy:.1f}")
            elif cmd == "m-":
                min_energy = max(0.0, min_energy - 5.0)
                print(f"Minimum energy: {min_energy:.1f}")
            
            elif cmd == "show":
                show_settings()
            elif cmd == "help":
                show_help()
            elif cmd == "midi":
                if out != None:
                    out.close()
                for i,p in enumerate(outports):
                    print(f"{i + 1}: {p}")
                output_port = input("Select new output device: ")
                out = mido.open_output(outports[int(output_port)-1])
                print(f"New output port: {out}")
            elif cmd:
                print("Unknown command. Type 'help' for available commands.")
                
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")


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
    input_devices = list_audio_devices()
    
    if not input_devices:
        print("ERROR: No input devices found!")
        sys.exit(1)
    
    while True:
        try:
            choice = input("Select device index: ")
            device_idx = int(choice)
            
            if device_idx in input_devices:
                device_info = sd.query_devices(device_idx)
                print(f"\nSelected: {device_info['name']}")
                print(f"Sample Rate: {int(device_info['default_samplerate'])} Hz")
                print(f"Available Channels: {device_info['max_input_channels']}")
                break
            else:
                print(f"Invalid device index. Please choose from: {input_devices}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    
    max_channels = device_info['max_input_channels']
    selected_channel = None
    
    if max_channels > 1:
        print(f"\nselected device has {max_channels} input channels")
        print("select channel/input audio source is connected to")
        
        for i in range(max_channels):
            print(f"  [{i+1}] Input {i+1} (Channel {i+1})")
        
        while True:
            try:
                channel_choice = input(f"\nSelect input (1-{max_channels}): ")
                channel_num = int(channel_choice)
                
                if 1 <= channel_num <= max_channels:
                    selected_channel = channel_num - 1
                    print(f"Using Input {channel_num}\n")
                    break
                else:
                    print(f"Please enter a number between 1 and {max_channels}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nExiting")
                sys.exit(0)
    else:
        selected_channel = 0
        print("Using single available input (Channel 0)\n")
    
    return device_idx, int(device_info['default_samplerate']), selected_channel


def find_dominant_frequency(audio_data, sample_rate):
    fft_data = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_data)
    peak_idx = np.argmax(magnitude)
    frequency = peak_idx * sample_rate / len(audio_data)
    
    return frequency, magnitude[peak_idx]

def calculate_band_energies(audio_data, sample_rate):
    fft_data = np.fft.rfft(audio_data)
    magnitude = np.abs(fft_data)
    freq_resolution = sample_rate / len(audio_data)
    bands = {
        'kick': (20, 1000),
        'snare': (1001, 3000),
        'hihat': (3000, 24000)
    }
    
    energies = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        low_bin = int(low_freq / freq_resolution)
        high_bin = int(high_freq / freq_resolution)
        
        high_bin = min(high_bin, len(magnitude))
        
        band_energy = np.sum(magnitude[low_bin:high_bin])
        energies[band_name] = band_energy
    
    return energies


def audio_callback_debug(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    if indata.shape[1] > 1:
        audio_data = indata[:, selected_channel]
    else:
        audio_data = indata[:, 0]
    audio_level = np.abs(audio_data).max()
    frequency, magnitude = find_dominant_frequency(audio_data, sample_rate)
    
    if audio_level > 0.001:
        print(f"Freq: {frequency:7.2f} Hz  |  Mag: {magnitude:10.2f}  |  Level: {audio_level:.4f}")


def audio_callback_drum(indata, frames, time_info, status):
    global last_detection, start_time
    
    if status:
        print(f"Status: {status}")
    
    if indata.shape[1] > 1:
        audio_data = indata[:, selected_channel]
    else:
        audio_data = indata[:, 0]
    
    audio_level = np.abs(audio_data).max()
    energies = calculate_band_energies(audio_data, sample_rate)
    current_time = time_info.currentTime if start_time is None else time_info.currentTime - start_time
    if start_time is None:
        start_time = time_info.currentTime
        current_time = 0
    
    kick_onset = energies['kick'] > kick_threshold
    snare_onset = energies['snare'] > snare_threshold
    hihat_onset = energies['hihat'] > hihat_threshold
    
    kick_debounced = (current_time - last_detection['kick']) > kick_debounce
    snare_debounced = (current_time - last_detection['snare']) > snare_debounce
    hihat_debounced = (current_time - last_detection['hihat']) > hihat_debounce
    
    total_energy = energies['kick'] + energies['snare'] + energies['hihat']
    
    if kick_onset and kick_debounced and total_energy > min_energy:
        if energies['kick'] / total_energy > 0.5:
            print(f"[KICK]    Time: {current_time:.3f}s  |  Freq Band: 20-1000Hz  |  Energy: {energies['kick']:.1f}  |  Level: {audio_level:.4f}")
            if out:
                out.send(mido.Message('note_on', note=60))
                out.send(mido.Message('note_off', note=60))
            last_detection['kick'] = current_time
    
    if snare_onset and snare_debounced and total_energy > min_energy:
        if energies['snare'] / total_energy > 0.3:
            print(f"[SNARE]   Time: {current_time:.3f}s  |  Freq Band: 1000-3000Hz  |  Energy: {energies['snare']:.1f}  |  Level: {audio_level:.4f}")
            if out:
                out.send(mido.Message('note_on', note=70))
                out.send(mido.Message('note_off', note=70))
            last_detection['snare'] = current_time
    
    if hihat_onset and hihat_debounced and total_energy > min_energy:
        if energies['hihat'] / total_energy > 0.4:
            print(f"[HI-HAT]  Time: {current_time:.3f}s  |  Freq Band: 3000Hz+  |  Energy: {energies['hihat']:.1f}  |  Level: {audio_level:.4f}")
            if out:
                out.send(mido.Message('note_on', note=80))
                out.send(mido.Message('note_off', note=80))
            last_detection['hihat'] = current_time


def audio_callback(indata, frames, time, status):
    if mode == "debug":
        audio_callback_debug(indata, frames, time, status)
    elif mode == "butcher":
        audio_callback_drum(indata, frames, time, status)


def main():
    global selected_device, sample_rate, selected_channel, mode
    
    print("\n" + "="*60)
    print("butcher")
    print("="*60)
    print("Press Ctrl+C to stop at any time")
    
    mode = select_mode()
    
    selected_device, sample_rate, selected_channel = select_device()
    
    print("Starting audio stream...")
    print("Butcher listening\n")
    
    if mode == "butcher":
        control_thread = threading.Thread(target=parameter_control_thread, daemon=True)
        control_thread.start()
    
    try:
        with sd.InputStream(
            device=selected_device,
            channels=2,
            samplerate=sample_rate,
            blocksize=BUFFER_SIZE,
            callback=audio_callback
        ):
            print("Press Ctrl+C to stop\n")
            while True:
                sd.sleep(1000)
                
    except KeyboardInterrupt:
        print("\n\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("Audio stream closed.")


if __name__ == "__main__":
    main()
