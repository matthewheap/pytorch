Use Case - I wanted to explore torch audio in an interactive sound-scape generation tool. This application leverages PyTorch and Torchaudio for high-performance DSP (Digital Signal Processing) and neural audio synthesis:

Neural Timbre (Audio Autoencoder): A 1-D Convolutional Neural Network (encoder/decoder) that creates unique "Neural Texture" layers. It compresses the audio into a latent representation and reconstructs it, introducing subtle, machine-learned timbral variations.
Spectral Feature Extraction: Uses Torchaudio to compute MFCCs (Mel-frequency cepstral coefficients) and Mel Spectrograms on the fly. These are used to analyze the spectral centroid, energy, and frequency profile of the recorded voice.
Intelligent Pitch Detection: Instead of a simple FFT, the app uses FFT-based Autocorrelation implemented in PyTorch tensors to accurately find the fundamental frequency of voiced (rather than sung) sounds.
Transient Onset Detection: Uses Spectral Flux (analyzing changes in frequency magnitude over time) to identify "onsets" (beats/transients) in your voice. These transients are then extracted and re-sequenced into rhythmic patterns.
Tensor-Based DSP: All audio transformations—including Pitch Shifting, Low-pass Biquad Filtering, and Granular Synthesis—are performed using PyTorch tensor operations, ensuring fast and efficient processing. 
To run, pull the code, install the requirements, and run app.py. Make sure you have a mic available.
