# TinyML-Based Voice Recognition System on Syntiant NDP101 - Thesis  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation code and documentation for my Bachelor's Degree final dissertation in **Computer, Communication and Electronic Engineering** at the **Department of Information Engineering and Computer Science**.

## Project Overview  
This thesis explores the implementation of a **TinyML voice recognition system** targeting **ultra-low-power embedded devices**, focusing on the **Syntiant NDP101 Neural Decision Processor**. The system combines **Keyword Spotting (KWS)** and **Speaker Verification (SV)** capabilities within the tight memory and computational constraints of edge devices.

### Objectives
- Implement **KWS** and **text-dependent SV** models suitable for deployment on the Syntiant NDP101.
- Develop a **C-based simulation pipeline** for audio processing and inference.
- Explore **distillation** and **quantization** techniques to compress deep learning models for deployment.
- Emulate **d-vector-based SV** in software due to NDA limitations on NDP101 SDK.

## System Architecture
### Hardware
![Uploading 4.05 Hardware Pipeline 2 NDP101.png…]()
- Dual **Syntiant NDP101** neural processors  
- **PDM microphone input** integrated in Syntiant
- SPI between master (KWS) and slave (SV) devices  

### Software Pipeline
1. **Signal Capture**: 968ms PDM audio, 15488 samples at 16kHz  
2. **Feature Extraction**: Log-mel spectrogram (40×40) using custom C implementation of Syntiant MFE  
3. **Keyword Spotting**: DNN model trained using Edge Impulse  
4. **Speaker Verification**:
   - CNN-based d-vector extractor (software emulated)
   - DNN-based distilled versions for deployment
   - Cosine similarity comparison
5. **Model Quantization**: Post-training and quantization-aware training (Int8, simulated Int4)

## Key Features

- **Keyword Spotting (KWS)** with Sheila keyword using DNN  
- **Speaker Verification (SV)** using d-vector extraction and cosine similarity  
- **Live and file-based inference** in pure C  
- **Model distillation** to create Syntiant-compatible DNN versions  
- **Quantization analysis** for Int8 and Int4 formats

## Results Summary
- Trained CNN-based SV model with 256-dimensional d-vectors  
- 5 distilled DNN variants tested for NDP101 compatibility  
- Functional C pipeline with simulated results close to theoretical targets  
- Quantized models show significant space reduction and maintain accuracy  

## Limitations
- **NDA restrictions** prevented actual deployment to NDP101  
- Syntiant supports **DNN only** (no CNN)  
- Int4 quantization is only partially supported without SDK tools  

## Future Work
- Deploy SV models on real hardware if SDK access is granted  
- Investigate adaptive or personalized training  
- Explore alternative platforms supporting CNNs on-device  
- Integration into smart systems like voice assistants or IoT devices  

## References
The project builds on:
- [Edge Impulse](https://www.edgeimpulse.com/) TinyML workflows  
- Syntiant NDP101 hardware documentation  
- TinySV and d-vector research  
- Audio preprocessing and feature extraction techniques  

For a full list of sources, see the bibliography section in the [thesis PDF](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis).

## 👤 Author
**Matteo Gottardelli**  
Bachelor’s Degree in Computer, Communication and Electronic Engineering  
Department of Information Engineering and Computer Science  
Academic Year 2024/2025

---

This project was developed as part of a Bachelor's final dissertation and represents a in-depth in edge AI and embedded systems.
