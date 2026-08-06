# virtual-orch

This is a repository for the VirtualOrch project, forked from the jam_bot (git repo jordanai).

In a nutshell, this app feeds in input from a MIDI keyboard through the following pipeline:

1. User input
2. MusicTransformer (this outputs piano music as of now. Aim: to output piano reductions of orchestral music.)
3. OrchestrationTransformer (this orchestrates the output from MusicTransformer)
4. route to virtual output port
5. which routes into Reaper.

This hopefully will support live orchestration of the piano inputs on top of the MusicTransformer outputs.

The transformers and playback lie on different JUCE threads. Informatino is passed from a thread to another through queues (CircularFIFO).