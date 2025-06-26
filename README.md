
# TinyML-Based Voice Recognition System on Syntiant NDP101 - From Keyword Spotting to Speaker Verification - Thesis   
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation code and documentation for my Bachelor's Degree final dissertation in **Computer, Communication and Electronic Engineering** at the **Department of Information Engineering and Computer Science** in Trento.

---

## Project Overview  
This thesis explores the implementation of a **TinyML voice recognition system** targeting **ultra-low-power embedded devices**, focusing on the **Syntiant NDP101 Neural Decision Processor**. The system combines **Keyword Spotting (KWS)** and text-dependent **Speaker Verification (SV)** capabilities within the tight memory and computational constraints of edge devices.

---

## Repo Structure (suggested)

```bash
Syntiant-NDP101-Speaker-Verification-Thesis/
├── models/
│   ├── kws_model.h
│   ├── sv_dnn_128.h
│   └── sv_cnn_reference/
├── src/
│   ├── main.c
│   ├── mfe.c
│   ├── kws.c
│   └── sv.c
├── data/
│   ├── sheila_ours.wav
│   └── sheila_google.wav
├── test/
│   └── results/
├── images/
│   └── *.png
└── README.md
```
To reproduce the code and the analysis, go to analysis folder and read the readME. This one is a general explanation of what achieved but not on how to make it work. 

## Syntiant Overview
![Syntiant Overview](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/5236d2728b30072128ad53e0efb447b9f43adcd1/images/2.01%20NDP101%20High%20Level%20Workflow.png)

The Syntiant NDP101 is neural decision processor designed for real-time, ultra-low power audio applications. It integrates:
   - Audio capture via dual **PDM microphone**\
   - A core NDP101 integrated with an Arduino Zero TinyML board
   - An **MFE (Mel-Feature Extractor)** hardware block
   - A **DNN inference** engine with limited support (no CNN)
   - SPI and GPIO interfaces for embedded communication

Despite powerful hardware features, its SDK is limited under NDA, restricting CNN-based models, requiring **software simulation** and **knowledge distillation**

## Objectives
- Build a **two-stage audio inference pipeline** (KWS → SV) optimized for Syntiant NDP101 **tiny embedded hardware**.
- Simulate audio signal capture, preprocessing, and model inference in pure **C**.
- Apply **d-vector extraction** and perform speaker verification using **cosine similarity**.
- Investigate and compare **knowledge distillation**, and **d-vector aggregation** methods to optimize performance and memory usage.
- Trying work around **Syntiant NDP101 SDK NDA restrictions** by creating a custom int-4 weights quantization (NOT SUCCESSED) -> Results were obtained in software simulation that reproduces hardware behavior.

---

## System Architecture

### Hardware Pipeline  

![Hardware Pipeline](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/5236d2728b30072128ad53e0efb447b9f43adcd1/images/4.05%20Hardware%20Pipeline%202%20NDP101.png)

Two **Syntiant NDP101** processors act as independent agents:
- **Master**: handles KWS
- **Slave**: handles SV (only triggers if KWS succeeds)

Communication is via **SPI**, using:
- MOSI (Master → Slave)
- MISO (Slave → Master)
- SCLK (Clock)
- CS/SS (Slave select)

This modular architecture allows scalable expansion by associating each microcontroller to a specific word-class (e.g., one NDP per command).

### Software Pipeline  

![Software Pipeline](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/5236d2728b30072128ad53e0efb447b9f43adcd1/images/4.01%20Software%20Pipeline.png)

The pipeline contains 5 major stages:

1. **Signal Capture**: 968ms of audio at 16kHz (15,488 samples)
2. **Feature Extraction**: Uses a **Log-Mel Spectrogram (40×40)** via:
   - **Pre-Emphasis**, boosting high frequencies to emphasize critical speech information
   - **Framing**, slicing audio into overlapping frames
   - **Hamming Window**, reducing spectral leakage
   - **FFT**, transforms signal into frequency domain (FFT via FFTW library)
   - **Mel Filterbank**, maps linear frequency spectrum to a Mel-scale (human perception)
   - **Log Energy + Flattening**, logarithmic compression and noise non-audible energy discarded
It generates a spectrogram 40x40.

![Feature Extraction](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/5236d2728b30072128ad53e0efb447b9f43adcd1/images/2.03%20MFE%20Block%20Processing.png)
   
3. **KWS Inference**:
    - Triggers SV only if keyword detected
    - Trained via Edge Impulse
    - Input Spectrogram 40x40
    - Structure FC256 -> FC256 -> FC256 -> Softmax 
    - Output belonging or not to a class (classification)
    - Fully compatible with Syntiant DNN constraints
  
![KWS inference](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/5236d2728b30072128ad53e0efb447b9f43adcd1/images/3.01%20KWS%20Model.png)
    
4. **SV Inference**:    
    - Trained with Tensorflow
    - Input Spectrogram 40x40
    - Model is a one-time trained, so it does not require retraining using reference d-vectors to differentiate people and words (text-dependency)
    - Model knowledge distilled from CNN -> DNN
    - Structure of CNN composed by Convolution 2D with BatchNorm and ReLu and Max-Pooling that gives a d-vector 128 or 256 size output
    - Various distillation DNN, consisting in 3 intermediate layers and one output and the 3 internal have varying neurons
    - Outputs a **d-vector**

![SV Inference](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/5236d2728b30072128ad53e0efb447b9f43adcd1/images/3.02%20D-vector%20Extractor.png)
5. **Cosine Similarity Output**:
    - Receives in input the d-vector
    - Compares it to stored references using **cosine similarity**.
    - The comparison is done only on reference in the words dominion given by KWS
    - The d-vector are stored in a permanent database
    
![Cosine Similarity Output](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/5236d2728b30072128ad53e0efb447b9f43adcd1/images/4.04%20D-Vector%20Processing.png)
6. **Enrollment**: Stores N reference vectors per user/word in memory (options: BEST, MEAN, GEOM MEDIAN)

---

## 🧱 Model In-depth

### KWS Model
Trained on ~3000 samples for "Sheila" and similar-sounding words.
Structure: [40x40]->FC256 -> FC256 -> FC256 -> Softmax("Sheila", "Other Word")

- Accuracy: 90.1%
- Precision: 97.9%
- Recall: 90.1%
- F1 Score: 93.9%
- EER: 10.1%
- AUC: 0.885

### SV Models  
- **CNN**: deeper semantics, fewer weights, not supported by Syntiant. Two models where explored:
- **DNN (Dense)**: deployable, higher memory, shallower understanding.
- Distilled using cosine similarity loss, achieving average similarity ~87.5%.

---

## Aggregation & Reference Management

| Method        | Accuracy | Memory  | Robustness | Speed |
|---------------|----------|---------|------------|-------|
| **BEST**      | High     | High    | Medium     | Fast |
| **GEOM MEDIAN** | Medium  | Medium  | High       | Slower |
| **MEAN**      | Lower    | Low     | Lower      | Fast |

---

## Results Summary

### KWS (Keyword Spotting)
- Accuracy: 90.1%
- Precision: 97.9%
- Recall: 90.1%
- EER: 10.1%
- AUC: 0.885

### SV with CNN
- Optimal at 64 references (recall ~94%, F1 ~0.82)

### SV with DNN
- Best deployable: 256-192 (F1 ~0.58, recall ~60%)
- Optimal memory trade-off: 128-128, Mean aggregation

---

## Trade-offs

| Factor                  | Trade-off |
|-------------------------|-----------|
| Accuracy vs Memory      | CNNs are best, but DNNs deployable |
| Recall vs Security      | Precision=1 reduces recall |
| Model Size vs Compatibility | 256 OK for sim, not NDP101 |
| Aggregation Method      | BEST accurate, MEAN smallest |

---

## Limitations
- CNNs not deployable due to SDK restriction
- No quantization tool access from Syntiant
- Float32 used in d-vectors for simulation

---

## Future Work
- Deploy with SDK once accessible
- Test adaptive SV training
- Port to boards with CNN support (e.g., NDP120)

---

## References

- Edge Impulse
- Google Speech Commands
- TinySV
- LibriSpeech
- Syntiant Docs

---

## Author

**Matteo Gottardelli**  
Bachelor’s Degree in Computer, Communication and Electronic Engineering  
Department of Information Engineering and Computer Science  
Academic Year 2024/2025  
Supervisor: Prof. Kasim Sinan Yildirim
Email: matteogottardelli@gmail.com
[GitHub Profile](https://github.com/Gotta003)

---

This project was developed as part of a Bachelor's final dissertation and represents a in-depth in edge AI and embedded systems.
