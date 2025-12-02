#!/usr/bin/env python3
from pythonosc import udp_client
import sounddevice as sd
import numpy as np
import sys
import random
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Static, Button, Label, RichLog, Input
from textual.reactive import reactive
from textual.binding import Binding

# random ass word list - change later
words = [
    "galaxy", "whisper", "cascade", "ember", "horizon",
    "shadow", "crystal", "thunder", "velvet", "marble",
    "spiral", "ocean", "midnight", "silver", "volcano",
    "eclipse", "diamond", "forest", "lightning", "phoenix",
    "cosmos", "glacier", "tempest", "crimson", "nebula",
    "amber", "thunder", "sapphire", "tornado", "zenith",
    "aurora", "bronze", "cavern", "delta", "emerald",
    "falcon", "granite", "harbor", "ivory", "jungle",
    "karma", "lava", "magnet", "nova", "onyx",
    "prism", "quartz", "raven", "storm", "tiger",
    "urban", "vortex", "winter", "xenon", "yellow",
    "zebra", "anchor", "blade", "cipher", "dragon",
    "echo", "flame", "ghost", "hammer", "iron",
    "javelin", "knight", "lotus", "meteor", "nectar",
    "orbit", "pulse", "quantum", "rocket", "solar",
    "titan", "ultra", "vertex", "wave", "xerox"
]

BUFFER_SIZE = 2048


class Butcher(App):
    """Butcher"""
    
    CSS = """
    Screen {
        background: $surface;
    }
    
    #output-container {
        height: 50%;
        border: solid $primary;
        margin: 1;
    }
    
    #controls-container {
        height: 50%;
        border: solid $accent;
        margin: 1;
        overflow-y: auto;
    }
    
    #controls-container > Horizontal {
        width: 100%;
        height: 100%;
    }
    
    RichLog {
        background: $surface-darken-1;
        color: $text;
        height: 100%;
    }
    
    .control-group {
        width: 1fr;
        height: 100%;
        border: solid $panel;
        margin: 0;
        padding: 0 1;
    }
    
    .control-row {
        width: 100%;
        height: auto;
        align: left middle;
        margin: 0 0 1 0;
    }
    
    .param-label {
        width: auto;
        min-width: 10;
        content-align: left middle;
    }
    
    Input {
        width: 12;
        height: 3;
    }
    
    .control-group > Label {
        margin: 0 0 1 0;
        text-style: bold;
    }
    
    .info-label {
        margin: 1 0;
        color: $text-muted;
    }
    
    .threshold-controls {
        background: $boost;
    }
    
    .debounce-controls {
        background: $boost;
    }
    
    .freq-controls {
        background: $boost;
    }
    
    .other-controls {
        background: $boost;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
    ]
    
    kick_threshold = reactive(100.0)
    snare_threshold = reactive(100.0)
    hihat_threshold = reactive(100.0)
    kick_debounce = reactive(0.12)
    snare_debounce = reactive(0.08)
    hihat_debounce = reactive(0.05)
    min_energy = reactive(10.0)
    velocity_multiplier = reactive(2.5)
    osc_ip = reactive("127.0.0.1")
    osc_port = reactive(9000)
    
    kick_freq_range = reactive((20, 150))
    snare_freq_range = reactive((150, 3000))
    hihat_freq_range = reactive((3000, 20000))
    
    def __init__(self):
        super().__init__()
        self.selected_device = None
        self.sample_rate = None
        self.selected_channel = None
        self.mode = None
        self.osc_client = None
        self.last_detection = {'kick': 0, 'snare': 0, 'hihat': 0}
        self.start_time = None
        self.stream = None
        
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        
        with Container(id="output-container"):
            yield RichLog(id="output", highlight=True, markup=True)
        
        with Container(id="controls-container"):
            with Horizontal():
                with Container(classes="control-group threshold-controls"):
                    yield Label("[b]THRESHOLDS[/b]")
                    with Horizontal(classes="control-row"):
                        yield Label("Kick:", classes="param-label")
                        yield Input(value=f"{self.kick_threshold:.1f}", id="kick_thresh_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Snare:", classes="param-label")
                        yield Input(value=f"{self.snare_threshold:.1f}", id="snare_thresh_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Hi-hat:", classes="param-label")
                        yield Input(value=f"{self.hihat_threshold:.1f}", id="hihat_thresh_input")
                
                with Container(classes="control-group debounce-controls"):
                    yield Label("[b]DEBOUNCE (ms)[/b]")
                    with Horizontal(classes="control-row"):
                        yield Label("Kick:", classes="param-label")
                        yield Input(value=f"{self.kick_debounce*1000:.0f}", id="kick_deb_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Snare:", classes="param-label")
                        yield Input(value=f"{self.snare_debounce*1000:.0f}", id="snare_deb_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Hi-hat:", classes="param-label")
                        yield Input(value=f"{self.hihat_debounce*1000:.0f}", id="hihat_deb_input")
                
                with Container(classes="control-group freq-controls"):
                    yield Label("[b]FREQ LOW (Hz)[/b]")
                    with Horizontal(classes="control-row"):
                        yield Label("Kick:", classes="param-label")
                        yield Input(value=f"{self.kick_freq_range[0]}", id="kick_freq_low_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Snare:", classes="param-label")
                        yield Input(value=f"{self.snare_freq_range[0]}", id="snare_freq_low_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Hi-hat:", classes="param-label")
                        yield Input(value=f"{self.hihat_freq_range[0]}", id="hihat_freq_low_input")
                
                with Container(classes="control-group freq-controls"):
                    yield Label("[b]FREQ HIGH (Hz)[/b]")
                    with Horizontal(classes="control-row"):
                        yield Label("Kick:", classes="param-label")
                        yield Input(value=f"{self.kick_freq_range[1]}", id="kick_freq_high_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Snare:", classes="param-label")
                        yield Input(value=f"{self.snare_freq_range[1]}", id="snare_freq_high_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Hi-hat:", classes="param-label")
                        yield Input(value=f"{self.hihat_freq_range[1]}", id="hihat_freq_high_input")
                
                with Container(classes="control-group other-controls"):
                    yield Label("[b]OTHER[/b]")
                    with Horizontal(classes="control-row"):
                        yield Label("Min Energy:", classes="param-label")
                        yield Input(value=f"{self.min_energy:.1f}", id="min_energy_input")
                    with Horizontal(classes="control-row"):
                        yield Label("Velocity:", classes="param-label")
                        yield Input(value=f"{self.velocity_multiplier:.1f}", id="vel_mult_input")
                    yield Label(f"OSC: {self.osc_ip}:{self.osc_port}", id="osc_label", classes="info-label")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the application"""
        output = self.query_one("#output", RichLog)
        output.write("[bold cyan]BUTCHER[/bold cyan]")
        output.write("[yellow]Initializing...[/yellow]")
        
        if self.mode == "butcher":
            output.write(f"[green]OSC configured: {self.osc_ip}:{self.osc_port}[/green]")
        output.write(f"[green]Mode: {self.mode}[/green]")
        output.write(f"[green]Device: {self.selected_device}, Sample Rate: {self.sample_rate}Hz[/green]")
        output.write("[bold green]Ready! Listening for audio...[/bold green]\n")
        
        self.start_audio_stream()
    
    def start_audio_stream(self):
        """Start the audio input stream"""
        try:
            self.stream = sd.InputStream(
                device=self.selected_device,
                channels=2,
                samplerate=self.sample_rate,
                blocksize=BUFFER_SIZE,
                callback=self.audio_callback
            )
            self.stream.start()
        except Exception as e:
            output = self.query_one("#output", RichLog)
            output.write(f"[bold red]Error starting audio stream: {e}[/bold red]")
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input field submissions"""
        input_id = event.input.id
        value_str = event.value
        
        try:
            if input_id == "kick_thresh_input":
                self.kick_threshold = max(10.0, float(value_str))
            elif input_id == "snare_thresh_input":
                self.snare_threshold = max(10.0, float(value_str))
            elif input_id == "hihat_thresh_input":
                self.hihat_threshold = max(10.0, float(value_str))
            
            elif input_id == "kick_deb_input":
                self.kick_debounce = max(0.02, min(0.30, float(value_str) / 1000.0))
            elif input_id == "snare_deb_input":
                self.snare_debounce = max(0.02, min(0.30, float(value_str) / 1000.0))
            elif input_id == "hihat_deb_input":
                self.hihat_debounce = max(0.02, min(0.30, float(value_str) / 1000.0))
            
            elif input_id == "kick_freq_low_input":
                low, high = self.kick_freq_range
                new_low = max(10, int(value_str))
                self.kick_freq_range = (min(new_low, high - 10), high)
            elif input_id == "snare_freq_low_input":
                low, high = self.snare_freq_range
                new_low = max(10, int(value_str))
                self.snare_freq_range = (min(new_low, high - 10), high)
            elif input_id == "hihat_freq_low_input":
                low, high = self.hihat_freq_range
                new_low = max(100, int(value_str))
                self.hihat_freq_range = (min(new_low, high - 100), high)
            
            elif input_id == "kick_freq_high_input":
                low, high = self.kick_freq_range
                new_high = min(20000, int(value_str))
                self.kick_freq_range = (low, max(new_high, low + 10))
            elif input_id == "snare_freq_high_input":
                low, high = self.snare_freq_range
                new_high = min(20000, int(value_str))
                self.snare_freq_range = (low, max(new_high, low + 10))
            elif input_id == "hihat_freq_high_input":
                low, high = self.hihat_freq_range
                new_high = min(20000, int(value_str))
                self.hihat_freq_range = (low, max(new_high, low + 100))
            
            elif input_id == "min_energy_input":
                self.min_energy = max(0.0, float(value_str))
            elif input_id == "vel_mult_input":
                self.velocity_multiplier = max(0.5, float(value_str))
                
        except ValueError:
            # if conversion fails, do not break my shit
            pass
    
    def watch_kick_threshold(self, value: float) -> None:
        try:
            self.query_one("#kick_thresh_input", Input).value = f"{value:.1f}"
        except:
            pass
    
    def watch_snare_threshold(self, value: float) -> None:
        try:
            self.query_one("#snare_thresh_input", Input).value = f"{value:.1f}"
        except:
            pass
    
    def watch_hihat_threshold(self, value: float) -> None:
        try:
            self.query_one("#hihat_thresh_input", Input).value = f"{value:.1f}"
        except:
            pass
    
    def watch_kick_debounce(self, value: float) -> None:
        try:
            self.query_one("#kick_deb_input", Input).value = f"{value*1000:.0f}"
        except:
            pass
    
    def watch_snare_debounce(self, value: float) -> None:
        try:
            self.query_one("#snare_deb_input", Input).value = f"{value*1000:.0f}"
        except:
            pass
    
    def watch_hihat_debounce(self, value: float) -> None:
        try:
            self.query_one("#hihat_deb_input", Input).value = f"{value*1000:.0f}"
        except:
            pass
    
    def watch_min_energy(self, value: float) -> None:
        try:
            self.query_one("#min_energy_input", Input).value = f"{value:.1f}"
        except:
            pass
    
    def watch_velocity_multiplier(self, value: float) -> None:
        try:
            self.query_one("#vel_mult_input", Input).value = f"{value:.1f}"
        except:
            pass
    
    def watch_kick_freq_range(self, value: tuple) -> None:
        try:
            self.query_one("#kick_freq_low_input", Input).value = f"{value[0]}"
            self.query_one("#kick_freq_high_input", Input).value = f"{value[1]}"
        except:
            pass
    
    def watch_snare_freq_range(self, value: tuple) -> None:
        try:
            self.query_one("#snare_freq_low_input", Input).value = f"{value[0]}"
            self.query_one("#snare_freq_high_input", Input).value = f"{value[1]}"
        except:
            pass
    
    def watch_hihat_freq_range(self, value: tuple) -> None:
        try:
            self.query_one("#hihat_freq_low_input", Input).value = f"{value[0]}"
            self.query_one("#hihat_freq_high_input", Input).value = f"{value[1]}"
        except:
            pass
    
    def find_dominant_frequency(self, audio_data):
        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        peak_idx = np.argmax(magnitude)
        frequency = peak_idx * self.sample_rate / len(audio_data)
        return frequency, magnitude[peak_idx]
    
    def calculate_band_energies(self, audio_data):
        fft_data = np.fft.rfft(audio_data)
        magnitude = np.abs(fft_data)
        freq_resolution = self.sample_rate / len(audio_data)
        bands = {
            'kick': self.kick_freq_range,
            'snare': self.snare_freq_range,
            'hihat': self.hihat_freq_range
        }
        
        energies = {}
        for band_name, (low_freq, high_freq) in bands.items():
            low_bin = int(low_freq / freq_resolution)
            high_bin = int(high_freq / freq_resolution)
            high_bin = min(high_bin, len(magnitude))
            band_energy = np.sum(magnitude[low_bin:high_bin])
            energies[band_name] = band_energy
        
        return energies
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback function"""
        if status:
            self.call_from_thread(self.log_output, f"[red]Status: {status}[/red]")
        
        if indata.shape[1] > 1:
            audio_data = indata[:, self.selected_channel]
        else:
            audio_data = indata[:, 0]
        
        if self.mode == "debug":
            self.process_debug_mode(audio_data)
        elif self.mode == "butcher":
            self.process_butcher_mode(audio_data, time_info)
    
    def process_debug_mode(self, audio_data):
        """Process audio in debug mode"""
        audio_level = np.abs(audio_data).max()
        frequency, magnitude = self.find_dominant_frequency(audio_data)
        
        if audio_level > 0.001:
            msg = f"Freq: {frequency:7.2f} Hz  |  Mag: {magnitude:10.2f}  |  Level: {audio_level:.4f}"
            self.call_from_thread(self.log_output, msg)
    
    def process_butcher_mode(self, audio_data, time_info):
        """Process audio in butcher mode"""
        audio_level = np.abs(audio_data).max()
        energies = self.calculate_band_energies(audio_data)
        
        current_time = time_info.currentTime if self.start_time is None else time_info.currentTime - self.start_time
        if self.start_time is None:
            self.start_time = time_info.currentTime
            current_time = 0
        
        kick_onset = energies['kick'] > self.kick_threshold
        snare_onset = energies['snare'] > self.snare_threshold
        hihat_onset = energies['hihat'] > self.hihat_threshold
        
        kick_debounced = (current_time - self.last_detection['kick']) > self.kick_debounce
        snare_debounced = (current_time - self.last_detection['snare']) > self.snare_debounce
        hihat_debounced = (current_time - self.last_detection['hihat']) > self.hihat_debounce
        
        total_energy = energies['kick'] + energies['snare'] + energies['hihat']
        
        if kick_onset and kick_debounced and total_energy > self.min_energy:
            if energies['kick'] / total_energy > 0.5:
                velocity = float(min(1.0, energies['kick'] / (self.kick_threshold * self.velocity_multiplier)))
                msg = f"[bold red]KICK[/bold red]    {self.kick_freq_range[0]}-{self.kick_freq_range[1]}Hz  |  Energy: {energies['kick']:.1f}  |  Vel: {velocity:.3f}"
                self.call_from_thread(self.log_output, msg)
                if self.osc_client:
                    word = random.choice(words)
                    self.osc_client.send_message("/butcher/kick", [word, velocity])
                self.last_detection['kick'] = current_time
        
        if snare_onset and snare_debounced and total_energy > self.min_energy:
            if energies['snare'] / total_energy > 0.3:
                velocity = float(min(1.0, energies['snare'] / (self.snare_threshold * self.velocity_multiplier)))
                msg = f"[bold yellow]SNARE[/bold yellow]   {self.snare_freq_range[0]}-{self.snare_freq_range[1]}Hz  |  Energy: {energies['snare']:.1f}  |  Vel: {velocity:.3f}"
                self.call_from_thread(self.log_output, msg)
                if self.osc_client:
                    word = random.choice(words)
                    self.osc_client.send_message("/butcher/snare", [word, velocity])
                self.last_detection['snare'] = current_time
        
        if hihat_onset and hihat_debounced and total_energy > self.min_energy:
            if energies['hihat'] / total_energy > 0.4:
                velocity = float(min(1.0, energies['hihat'] / (self.hihat_threshold * self.velocity_multiplier)))
                msg = f"[bold cyan]HI-HAT[/bold cyan]  {self.hihat_freq_range[0]}-{self.hihat_freq_range[1]}Hz  |  Energy: {energies['hihat']:.1f}  |  Vel: {velocity:.3f}"
                self.call_from_thread(self.log_output, msg)
                if self.osc_client:
                    word = random.choice(words)
                    self.osc_client.send_message("/butcher/hihat", [word, velocity])
                self.last_detection['hihat'] = current_time
    
    def log_output(self, message: str) -> None:
        """Log a message to the output window"""
        output = self.query_one("#output", RichLog)
        output.write(message)
    
    def action_quit(self) -> None:
        """Quit the application"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.exit()


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


def configure_osc():
    print("\n" + "="*60)
    print("OSC CONFIGURATION")
    print("="*60)
    
    osc_ip = "127.0.0.1"
    osc_port = 9000
    
    ip_input = input(f"Enter OSC IP address (default: {osc_ip}): ").strip()
    if ip_input:
        osc_ip = ip_input
    
    port_input = input(f"Enter OSC port (default: {osc_port}): ").strip()
    if port_input:
        osc_port = int(port_input)
    
    osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)
    
    print(f"\nOSC configured to send to: {osc_ip}:{osc_port}")
    print("Messages will be sent to:")
    print("  /butcher/kick [word, velocity]")
    print("  /butcher/snare [word, velocity]")
    print("  /butcher/hihat [word, velocity]")
    print()
    
    return osc_client, osc_ip, osc_port


def main():
    print("\n" + "="*60)
    print("butcher")
    print("="*60)
    print("Press Ctrl+C to stop at any time")
    
    mode = select_mode()
    selected_device, sample_rate, selected_channel = select_device()
    
    osc_client = None
    osc_ip = "127.0.0.1"
    osc_port = 9000
    
    if mode == "butcher":
        osc_client, osc_ip, osc_port = configure_osc()
    
    app = Butcher()
    app.mode = mode
    app.selected_device = selected_device
    app.sample_rate = sample_rate
    app.selected_channel = selected_channel
    app.osc_client = osc_client
    app.osc_ip = osc_ip
    app.osc_port = osc_port
    
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        print("Audio stream closed.")


if __name__ == "__main__":
    main()
