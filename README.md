# TinyML-Based Voice Recognition on Syntiant NDP101

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation code and documentation for my Bachelor's Degree final dissertation in Computer, Communication and Electronic Engineering at the Department of Information Engineering and Computer Science.

## Project Overview

This project explores the implementation of voice recognition on ultra-low-power neural network accelerators, specifically the Syntiant NDP101 chip. The work extends beyond basic Keyword Spotting (KWS) to include Speaker Verification (SV) capabilities, creating a complete voice authentication system on a TinyML platform.

### Key Features

- **Keyword Spotting (KWS)**: Implementation using Edge Impulse with full C integration
- **Speaker Verification (SV)**: Custom d-vector extractor approach based on TinySV principles
- **Model Quantization**: Systematic approach for 8-bit quantization with comparative analysis
- **Complete Pipeline**: From audio capture to neural network processing and verification

## System Architecture

The system consists of:
- ESP32 microcontroller
- 2x Syntiant NDP101 neural processors
- PDM microphone input

### Hardware Pipeline
```
 [ESP32] ↔ [Syntiant NDP101] ↔ [Syntiant NDP101]
(Control)      (KWS Model)         (SV Model)
```

### Software Pipeline
1. **Signal Capture**: Direct PDM sampling on NDP101 (15488 samples)
2. **MFE Block Generation**: Mel filterbank feature extraction (1600 features)
3. **KWS Model**: FC256→FC256→FC256→Softmax(4)
4. **SV Model**: CNN with batch normalization producing 256-dimensional d-vectors

## Implementation Details

### Keyword Spotting Model
- Trained using Edge Impulse
- Custom C implementation for direct MFE computation on device
- Softmax classification for target keywords

### Speaker Verification Model
- D-vector extraction using CNN architecture
- Database design for reference sample storage
- Cosine similarity calculation for speaker matching
- Two verification methods: best-matching and mean-cosine

### Quantization Process
- Float32 → Int8 conversion with minimal accuracy loss
- Challenges with Int4 quantization on TinyML platforms
- Performance comparisons between original and quantized models

## Results

The repository includes:
- KWS performance analysis (confusion matrices)
- SV performance comparison between float32 and int8 precision
- System integration metrics (power consumption, memory usage)
- Trade-off analysis between accuracy and resource constraints

## Key Contributions

1. Full C implementation of MFE/DNN processing for accurate simulation
2. Distilled Speaker Verification model suitable for TinyML applications
3. Systematic approach to model quantization for edge deployment

## Technical Limitations

- SDK barriers due to NDA restrictions
- Hardware constraints of DNN-only architecture
- Quantization challenges at 4-bit precision

## Future Work

- Deployment of binary SV model with Syntiant SDK access
- Integration with existing voice-activated systems
- Adaptive training capabilities for personalized voice recognition

## Getting Started

### Prerequisites
- Node.js ([deb.nodesource.com](http://deb.nodesource.com))
- ESP32 development environment
- Edge Impulse CLI tools

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/TinyML-Voice-Recognition-NDP101.git

# Install dependencies
npm install

# Set up Edge Impulse CLI (if needed)
npm install -g edge-impulse-cli
```

## References

The project builds upon work from:

- Edge Impulse TinyML deployments
- Syntiant NDP101 documentation
- TinySV approach for lightweight speaker verification
- Various papers on MFE and audio processing for TinyML

For a complete list of references, see the bibliography section in the thesis document.

## Author

**Matteo Gottardelli**  
Department of Information Engineering and Computer Science

---

*This project was developed as part of a Bachelor's Degree final dissertation, 2024/2025.*
