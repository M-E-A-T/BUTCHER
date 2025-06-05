




s.waitForBoot({

	// A very simple example that should work without errors
{
// Path to your audio file (update this to your actual file path)
~path = "/Users/dudu/dev/supercollider/Krack.mp3";

// Load the file into a buffer
~buffer = Buffer.read(s, ~path);

// Create a simple analysis synth with fixed bus indices
SynthDef(\simpleAnalyzer, {
    // PlayBuf to read from buffer
    var in = PlayBuf.ar(2, ~buffer, BufRateScale.kr(~buffer), loop: 1);
    var mono = Mix(in) / 2;

    // Simple analysis with fixed bus numbers
    var low = Amplitude.kr(LPF.ar(mono, 200));
    var mid = Amplitude.kr(BPF.ar(mono, 1000, 1));
    var high = Amplitude.kr(HPF.ar(mono, 5000));

    // Send to fixed bus numbers (10-12)
/*    Out.kr(10, low);
    Out.kr(11, mid);
    Out.kr(12, high);*/
	Poll.kr(Impulse.kr(4), low, "Bass");
    Poll.kr(Impulse.kr(4), mid, "Mid");
    Poll.kr(Impulse.kr(4), high, "High");

    // Output audio
    Out.ar(0, in);
}).add;
	}.();


	{
		s.sync;
	}.();

	{
		// Start the analyzer
		~analyzer = Synth(\simpleAnalyzer);
	}.();


})