# Software Design Document for Butcher - Converting music into meaningful data
- Khalid Diab
- Orlando Kenny
- Alex Suarez
- Fiona Kelly
- Ishaan

# Overview
The Butcher, a lightweight, realtime program capable of transforming audio input into meaningful data structures that can be used to control external devices.

# Context
Musical visuals usually SUCK at live music events, they're either not on beat or have nothing to do with the song being played. The Butcher aims to allow enthusiasts to use live data from any audio source to control whatever they want with it. Our strategy is to transform an incoming audio signal into an easy-to-use data structure that can be used to infer something about the audio signal at hand.

# Goals
Empowering people to control anything using a lightweight device and an audio signal.

## Success Metrics
- Audio signal is translated in real time.
- Butcher data is sent to user in real time.
- User can connect and receive data from the Butcher.

# Technologies
- supercollider
- raspberry pi

# Methodology
Supercollider or more specifically sclang will be used to calculate the items we want to include in our data structres, using sclang will keep the program lightweight and fast. We might need to use tensorflow or python for the machine learning algorithms but ideally C should be used to keep it fast.

## FFT
Using FFT we can extract a lot of information out of an audio signal, the Butcher will focus on the following:
- Musical notes being played (This is obtained by normalizing the frequencies and assigning a note value to it)
- Chords (Inferred from the notes being played)
- Key (Inferred from the notes being played)
- Bass hits (This can be obtained by checking a specific frequency range)
- Hi hats (This can be obtained by checking a specific frequency range)
- Mood (Can be inferred from key, although we would like to use a ML algorithm to decide the genre)

## ML
- Mood (Ideas for this include using a buffer to calculate the mood of the last 10 seconds worth of data points, the faster the better)
- Genre (ML Algorithm to find out the music genre, might use the same buffer idea for this aswell)

# Solutions & Strategies
## Drum Hit Detection
### Phase 1 (Drum Template Creation):
1. Collect Drum samples
- Record high quality samples of individual kick, snare and hi-hat sounds
- use multiple variations of each (I'm thinking 30 each)
2. Generate Spectograms
- Process each sample using STFT (short-time fourier transform)
- Use consistent params (2048 window size, 512 hop size)
- Save magnitude spectograms (discard phase info)
3. Create Representative Templates
- Average spectograms for each drum type
- Apply Characteristic frequency filters (kick: 50-200Hz, snare: 200-1200Hz, hi-hat: 3000-16000Hz)
- Normalize for consistent comparision
4. Export to Supercollider
- Save as floating-point array or WAV files that SC can read and include metadata

### Phase 2 (Supercollider Implementation):
1. Load Templates
- Import pre-computed templates into SC buffers at initialization
- Verify correct loading by checking buffer properties
2. Set Up Real-Time Analysis
- Create FFT analysis chain for incoming audio
- Configure FFT size to match template dimensions
- Set up spectral processing modules
3. Implement Detection Algorithm
- Calculate similarity between current audio spectrogram and templates
- Use specialized distance metrics (as described in Dittmar's paper)
- Apply dynamic thresholding for detection decisions
4. Output Results
- Generate trigger signals when drums are detected
- Calculate confidence levels based on similarity scores
- Optionally, feed results to SC patterns or synthesis modules
### Phase 3 (Optimize):
idk 

## Practical Workflow
1. Template Creation
- Create templates in Python/MATLAB and export as WAV files
2. Initialize SC System
- Boot server and load template buffers
- Configure analysis parameters
3. Process Audio Stream
- Perform FFT analysis on incoming audio
- Calculate similarity metrics against templates
- Apply thresholds to determine detections
4. Generate Musical Events
- Convert detections to MIDI or OSC messages
- Trigger synthesis or control other processes
