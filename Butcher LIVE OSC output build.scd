s.boot;

(
// Define external OSC target (edit IP/port if needed)
~target = NetAddr("127.0.0.1", 9000);
)

(
// SynthDef for live audio drum detection with internal + external OSC
SynthDef(\drumDetectorLive, {
    arg threshold = 0.23;

    var input, kickBand, snareBand, hihatBand;
    var kickEnv, snareEnv, hihatEnv;
    var kickTrig, snareTrig, hihatTrig;
    var kickLevel, snareLevel, hihatLevel;
    var zeroCrossings, currentTime;
    var fft, spectralCentroid;
    var totalEnergy;

    input = Mix(SoundIn.ar([0, 1]));
    input = Compander.ar(input, input, thresh: -20, slopeBelow: 0, slopeAbove: 1, clampTime: 0.01, relaxTime: 0.1);

    currentTime = Sweep.ar(0, 1);

    kickBand = LPF.ar(input, 100);
    snareBand = BPF.ar(input, 350, 2.0);
    hihatBand = HPF.ar(input, 8000);

    kickEnv = Amplitude.ar(kickBand, 0.01, 0.1);
    snareEnv = Amplitude.ar(snareBand, 0.01, 0.1);
    hihatEnv = Amplitude.ar(hihatBand, 0.001, 0.05);
    totalEnergy = Amplitude.ar(input, 0.01, 0.1);
    zeroCrossings = ZeroCrossing.ar(input);
    fft = FFT(LocalBuf(1024), input);
    spectralCentroid = SpecCentroid.kr(fft).clip(20, 20000);

    kickTrig = Trig1.ar((kickEnv > (threshold * 1)) * (kickEnv > (snareEnv + 0.01)) * (kickEnv > (hihatEnv * 2)), 0.15);
    snareTrig = Trig1.ar((snareEnv > threshold * 1.8) * (totalEnergy > (threshold * 0.5)) * (snareEnv > (kickEnv * 0.4)), 0.1);
    hihatTrig = Trig1.ar((hihatEnv > (threshold * 0.3)) * (hihatEnv > (kickEnv * 1.5)) * (zeroCrossings > 0.1), 0.06);

    kickLevel = Latch.ar(kickEnv, kickTrig);
    snareLevel = Latch.ar(snareEnv, snareTrig);
    hihatLevel = Latch.ar(hihatEnv, hihatTrig);

    // SendReply for internal logging
    SendReply.ar(kickTrig, '/kick', [currentTime, kickLevel, Latch.ar(spectralCentroid, kickTrig), Latch.ar(zeroCrossings, kickTrig), kickEnv / (snareEnv + 0.001), kickEnv / (hihatEnv + 0.001)]);
    SendReply.ar(snareTrig, '/snare', [currentTime, snareLevel, Latch.ar(spectralCentroid, snareTrig), Latch.ar(zeroCrossings, snareTrig), snareEnv / (kickEnv + 0.001), snareEnv / (hihatEnv + 0.001)]);
    SendReply.ar(hihatTrig, '/hihat', [currentTime, hihatLevel, Latch.ar(spectralCentroid, hihatTrig), Latch.ar(zeroCrossings, hihatTrig), hihatEnv / (kickEnv + 0.001), hihatEnv / (snareEnv + 0.001)]);

    // SendTrig for external OSC
    SendTrig.ar(kickTrig, 11, currentTime);
    SendTrig.ar(kickTrig, 12, kickLevel);
    SendTrig.ar(kickTrig, 13, Latch.ar(spectralCentroid, kickTrig));
    SendTrig.ar(kickTrig, 14, Latch.ar(zeroCrossings, kickTrig));
    SendTrig.ar(kickTrig, 15, kickEnv / (snareEnv + 0.001));
    SendTrig.ar(kickTrig, 16, kickEnv / (hihatEnv + 0.001));

    SendTrig.ar(snareTrig, 21, currentTime);
    SendTrig.ar(snareTrig, 22, snareLevel);
    SendTrig.ar(snareTrig, 23, Latch.ar(spectralCentroid, snareTrig));
    SendTrig.ar(snareTrig, 24, Latch.ar(zeroCrossings, snareTrig));
    SendTrig.ar(snareTrig, 25, snareEnv / (kickEnv + 0.001));
    SendTrig.ar(snareTrig, 26, snareEnv / (hihatEnv + 0.001));

    SendTrig.ar(hihatTrig, 31, currentTime);
    SendTrig.ar(hihatTrig, 32, hihatLevel);
    SendTrig.ar(hihatTrig, 33, Latch.ar(spectralCentroid, hihatTrig));
    SendTrig.ar(hihatTrig, 34, Latch.ar(zeroCrossings, hihatTrig));
    SendTrig.ar(hihatTrig, 35, hihatEnv / (kickEnv + 0.001));
    SendTrig.ar(hihatTrig, 36, hihatEnv / (snareEnv + 0.001));

    Out.ar(0, [input, input] * 0.5);
}).add;
)

(
// OSCDef responders for internal logging
OSCdef(\kickResponder, { |msg|
    var t = msg[3], level = msg[4], c = msg[5], z = msg[6], d1 = msg[7], d2 = msg[8];
    "KICK @ %.2f | Lvl: %.2 | Centroid: %Hz | ZCR: %.2 | Dom: %.1 | Hhat: %.1"
    .format(t, level, c.round(1), z, d1, d2).postln;
}, '/kick');

OSCdef(\snareResponder, { |msg|
    var t = msg[3], level = msg[4], c = msg[5], z = msg[6];
    "SNARE @ %.2f | Lvl: %.2 | Centroid: %Hz | ZCR: %.2"
    .format(t, level, c.round(1), z).postln;
}, '/snare');

OSCdef(\hihatResponder, { |msg|
    var t = msg[3], level = msg[4], c = msg[5], z = msg[6];
    "HIHAT @ %.2f | Lvl: %.2 | Centroid: %Hz | ZCR: %.2"
    .format(t, level, c.round(1), z).postln;
}, '/hihat');
)

(
// External OSC forwarder using SendTrig IDs
OSCFunc({ |msg|
    var id = msg[2], val = msg[3];
    switch(id,
        11, { ~target.sendMsg("/kick/time", val) },
        12, { ~target.sendMsg("/kick/level", val) },
        13, { ~target.sendMsg("/kick/centroid", val) },
        14, { ~target.sendMsg("/kick/zcr", val) },
        15, { ~target.sendMsg("/kick/vsSnare", val) },
        16, { ~target.sendMsg("/kick/vsHihat", val) },

        21, { ~target.sendMsg("/snare/time", val) },
        22, { ~target.sendMsg("/snare/level", val) },
        23, { ~target.sendMsg("/snare/centroid", val) },
        24, { ~target.sendMsg("/snare/zcr", val) },
        25, { ~target.sendMsg("/snare/vsKick", val) },
        26, { ~target.sendMsg("/snare/vsHihat", val) },

        31, { ~target.sendMsg("/hihat/time", val) },
        32, { ~target.sendMsg("/hihat/level", val) },
        33, { ~target.sendMsg("/hihat/centroid", val) },
        34, { ~target.sendMsg("/hihat/zcr", val) },
        35, { ~target.sendMsg("/hihat/vsKick", val) },
        36, { ~target.sendMsg("/hihat/vsSnare", val) }
    );
}, '/tr');
)

(
// Start detector
~liveDetector = Synth(\drumDetectorLive, [\threshold, 0.23]);
)

// ~liveDetector.set(\threshold, 0.13);
// ~liveDetector.free;
