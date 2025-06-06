s.boot;


(
~buffer = Buffer.read(s, "path");
)

(
SynthDef(\drumDetectorFile, {
    arg bufnum, threshold = 0.4, rate = 1.0, startPos = 0;

    var input, kickBand, snareBand, hihatBand;
    var kickEnv, snareEnv, hihatEnv;
    var kickTrig, snareTrig, hihatTrig;
    var kickLevel, snareLevel, hihatLevel;
    var zeroCrossings, currentTime;
    var fft, spectralCentroid;
    var totalEnergy;

    // Play the audio file
    input = PlayBuf.ar(1, bufnum, rate, startPos: startPos, loop: 0, doneAction: 2);

    // Track current playback time for timestamps
    currentTime = Phasor.ar(0, rate / SampleRate.ir, startPos, BufFrames.kr(bufnum));

    // Create frequency bands
    kickBand = LPF.ar(input, 80);           // Bass: 20-80Hz
    snareBand = BPF.ar(input, 250, 2.0);    // Snare: ~150-400Hz
    hihatBand = HPF.ar(input, 8000);        // Hi-hat: 8kHz+

    // Envelope followers for each band
    kickEnv = Amplitude.ar(kickBand, 0.01, 0.1);
    snareEnv = Amplitude.ar(snareBand, 0.01, 0.1);
    hihatEnv = Amplitude.ar(hihatBand, 0.001, 0.05);
    totalEnergy = Amplitude.ar(input, 0.01, 0.1);

    // Calculate basic features
    zeroCrossings = ZeroCrossing.ar(input);

    // FFT analysis for spectral centroid (control rate for efficiency)
    fft = FFT(LocalBuf(1024), input);
    spectralCentroid = SpecCentroid.kr(fft).clip(20, 20000);

	//trigger detection
    kickTrig = Trig1.ar(
        (kickEnv > (threshold * 2)) *
        (kickEnv > (snareEnv + 0.01)) *     // Kick should dominate over snare
        (kickEnv > (hihatEnv * 2)),         // Kick should dominate over hihat
        0.15  // Longer holdoff to prevent multiple triggers
    );

    snareTrig = Trig1.ar(
        (snareEnv > threshold) *
        (totalEnergy > (threshold * 0.5)) * // General activity threshold
        (snareEnv > (kickEnv * 0.4)),       // Some independence from kick
        0.1
    );

    hihatTrig = Trig1.ar(
        (hihatEnv > (threshold * 0.3)) *
        (hihatEnv > (kickEnv * 1.5)) *     // Hi-hat should be louder than kick
        (zeroCrossings > 0.1),             // High-frequency content indicator
        0.06 //shroter holdoff
    );

    // Get current levels and features when triggered
    kickLevel = Latch.ar(kickEnv, kickTrig);
    snareLevel = Latch.ar(snareEnv, snareTrig);
    hihatLevel = Latch.ar(hihatEnv, hihatTrig);

    // Send OSC messages with timestamp and analysis data
    SendReply.ar(kickTrig, '/kick', [
        currentTime / SampleRate.ir,        // Timestamp in seconds
        kickLevel,
        Latch.ar(spectralCentroid, kickTrig),
        Latch.ar(zeroCrossings, kickTrig),
        kickEnv / (snareEnv + 0.001),       // Kick dominance ratio
        kickEnv / (hihatEnv + 0.001)        // Kick vs hihat ratio
    ]);

    SendReply.ar(snareTrig, '/snare', [
        currentTime / SampleRate.ir,
        snareLevel,
        Latch.ar(spectralCentroid, snareTrig),
        Latch.ar(zeroCrossings, snareTrig),
        snareEnv / (kickEnv + 0.001),       // Snare vs kick ratio
        snareEnv / (hihatEnv + 0.001)       // Snare vs hihat ratio
    ]);

    SendReply.ar(hihatTrig, '/hihat', [
        currentTime / SampleRate.ir,
        hihatLevel,
        Latch.ar(spectralCentroid, hihatTrig),
        Latch.ar(zeroCrossings, hihatTrig),
        hihatEnv / (kickEnv + 0.001),       // Hihat vs kick ratio
        hihatEnv / (snareEnv + 0.001)       // Hihat vs snare ratio
    ]);

    // Output original signal for monitoring
    Out.ar(0, input * 0.5);

}).add;
)

// Set up OSC responders with detailed analysis
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

// Start analyzing the file
(
~detector = Synth(\drumDetectorFile, [\bufnum, ~buffer]);
)

// Clean up stuff
// ~detector.free;
// ~detector = Synth(\drumDetectorFile, [\bufnum, ~buffer, \startPos, 30 * 44100]);

~detector.set(\threshold, 0.4);

~detector.free;
s.freeAll


(
OSCdef(\kickResponder).free;
OSCdef(\snareResponder).free;
OSCdef(\hihatResponder).free;
~buffer.free;
)