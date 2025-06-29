s.boot;

(
SynthDef(\drumDetectorLive, {
    arg in = 0, threshold = 0.23;  // 'in' specifies which audio input (0 = first input)

    var input, kickBand, snareBand, hihatBand;
    var kickEnv, snareEnv, hihatEnv;
    var kickTrig, snareTrig, hihatTrig;
    var kickLevel, snareLevel, hihatLevel;
    var zeroCrossings, currentTime;
    var fft, spectralCentroid;
    var totalEnergy;

    // Get live audio input

	input = Mix(SoundIn.ar([0, 1]));  // sum stereo input to mon
	input = Compander.ar(input, input, thresh: -20, slopeBelow: 0, slopeAbove: 1, clampTime: 0.01, relaxTime: 0.1);


    // running timer for timestamps
    currentTime = Sweep.ar(0, 1);  // Counts seconds since synth started

    // Create frequency bands
    kickBand = LPF.ar(input, 100);           // Bass: 20-80Hz
    snareBand = BPF.ar(input, 350, 2.0);    // Snare: ~150-400Hz
    hihatBand = HPF.ar(input, 8000);        // Hi-hat: 8kHz+

    // Envelope followers for each band
    kickEnv = Amplitude.ar(kickBand, 0.01, 0.1);
    snareEnv = Amplitude.ar(snareBand, 0.01, 0.1);
    hihatEnv = Amplitude.ar(hihatBand, 0.001, 0.05);
    totalEnergy = Amplitude.ar(input, 0.01, 0.1);

    // Calculate basic features
    zeroCrossings = ZeroCrossing.ar(input);

    // FFT analysis for spectral centroid
    fft = FFT(LocalBuf(1024), input);
    spectralCentroid = SpecCentroid.kr(fft).clip(20, 20000);

    // Trigger detection
    kickTrig = Trig1.ar(
		(kickEnv > (threshold * 4)) *
        (kickEnv > (snareEnv + 0.01)) *
        (kickEnv > (hihatEnv * 2)),
        0.15
    );

    snareTrig = Trig1.ar(
        (snareEnv > threshold * 1.8) *
        (totalEnergy > (threshold * 0.5)) *
        (snareEnv > (kickEnv * 0.4)),
        0.1
    );

    hihatTrig = Trig1.ar(
        (hihatEnv > (threshold * 0.3)) *
        (hihatEnv > (kickEnv * 1.5)) *
        (zeroCrossings > 0.1),
        0.06
    );

    // Get current levels and features when triggered
    kickLevel = Latch.ar(kickEnv, kickTrig);
    snareLevel = Latch.ar(snareEnv, snareTrig);
    hihatLevel = Latch.ar(hihatEnv, hihatTrig);

    // Send OSC messages with timestamp and analysis data
    SendReply.ar(kickTrig, '/kick', [
        currentTime,
        kickLevel,
        Latch.ar(spectralCentroid, kickTrig),
        Latch.ar(zeroCrossings, kickTrig),
        kickEnv / (snareEnv + 0.001),
        kickEnv / (hihatEnv + 0.001)
    ]);

    SendReply.ar(snareTrig, '/snare', [
        currentTime,
        snareLevel,
        Latch.ar(spectralCentroid, snareTrig),
        Latch.ar(zeroCrossings, snareTrig),
        snareEnv / (kickEnv + 0.001),
        snareEnv / (hihatEnv + 0.001)
    ]);

    SendReply.ar(hihatTrig, '/hihat', [
        currentTime,
        hihatLevel,
        Latch.ar(spectralCentroid, hihatTrig),
        Latch.ar(zeroCrossings, hihatTrig),
        hihatEnv / (kickEnv + 0.001),
        hihatEnv / (snareEnv + 0.001)
    ]);

    // Output original signal for monitoring
    Out.ar(0, [input, input] * 0.5);

}).add;
)

// Set up OSC responders (same as before)
(
OSCdef(\kickResponder, {
    arg msg, time;
    var timestamp = msg[3];
    var level = msg[4];
    var centroid = msg[5];
    var zcr = msg[6];
    var kickDominance = msg[7];
    var kickVsHihat = msg[8];

    "KICK @ %s - Level: %, Centroid: %Hz, Dominance: %".postf(
        timestamp.round(0.01),
        level.round(0.03),
        centroid.round(1),
        kickDominance.round(0.1)
    );
    "".postln;
}, '/kick');
)
(
OSCdef(\snareResponder, {
    arg msg, time;
    var timestamp = msg[3];
    var level = msg[4];
    var centroid = msg[5];
    var zcr = msg[6];
    var snareVsKick = msg[7];
    var snareVsHihat = msg[8];

    "SNARE @ %s - Level: %, Centroid: %Hz, ZCR: %".postf(
        timestamp.round(0.01),
        level.round(0.03),
        centroid.round(1),
        zcr.round(0.01)
    );
    "".postln;
}, '/snare');
)
(
OSCdef(\hihatResponder, {
    arg msg, time;
    var timestamp = msg[3];
    var level = msg[4];
    var centroid = msg[5];
    var zcr = msg[6];
    var hihatVsKick = msg[7];
    var hihatVsSnare = msg[8];

    "HI-HAT @ %s - Level: %, Centroid: %Hz, ZCR: %".postf(
        timestamp.round(0.01),
        level.round(0.03),
        centroid.round(1),
        zcr.round(0.01)
    );
    "".postln;
}, '/hihat');
)

// Start the live detector
(
~liveDetector = Synth(\drumDetectorLive, [
    \in, 1, // <- this will use the 2nd audio device, change if needed
    \threshold, 0.23
]);
)

// Adjust threshold
~liveDetector.set(\threshold, 0.2 );

// Change input channel
~liveDetector.set(\in, 1);

// Kill detector
~liveDetector.free;
