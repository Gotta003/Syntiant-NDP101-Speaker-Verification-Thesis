
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

## Models

Structure:
| Model Name | KWS     | SV128      | SV256      | SVD128 | SVDU256 | SVD192 | SVD240 | SVD2256 |
|------------|---------|------------|------------|--------|---------|--------|--------|---------|
| Type       | Dense   | Conv       | Conv       | Dense  | Dense   | Dense  | Dense  | Dense   |
| Origin     | -       | -          | -          | SV128  | SV256   | SV256  | SV256  | SV256   |
| Input      | 1600    | (40x40x1)  | (40x40x1)  | 1600   | 1600    | 1600   | 1600   | 1600    |
| Layer 1    | 256     | (13x13x8)  | (13x13x8)  | 256    | 256     | 192    | 240    | 256     |
| Layer 2    | 256     | (6x6x16)   | (6x6x16)   | 256    | 256     | 192    | 240    | 256     |
| Layer 3    | 256     | (3x3x32)   | (3x3x32)   | 256    | 128     | 192    | 240    | 256     |
| Layer 4    | -       | (2x2x64)   | -          | -      | -       | -      | -      | -       |
| Layer Out  | 2       | 128        | 256        | 128    | 256     | 256    | 256    | 256     |
| Total (KB) | 2117    | 383,6      | 95,2       | 2243,5 | 2115,5  | 1683,25| 2193,8 | 2372    |
| Deployable | YES     | NO         | NO         | YES    | YES     | YES    | YES    | NO      |

Capability in dataset:

| Type               | D-vector = 128                              |         |         |         |       |  |
|--------------------|---------------------------------------------|---------|---------|---------|------------|-------------|
| Aggregation        | Best                                        |         |         |         | Mean           | Geom_Median  |
| N° Refs            | 1         | 8       | 16      | 64      | All        | All         |
| Size of Word (B)   | 512      | 4096    | 8192    | 32768   | 512        | 512         |
| Quant (B)          | 64       | 512     | 1024    | 4096    | 64         | 64          |
| Words Float        | 128      | 16      | 8       | 2       | 128        | 128         |
| Words 4-int        | 1024     | 128     | 64      | 16      | 1024       | 1024        |

| Type               | D-vector = 256                              |         |         |         |        |  |
|--------------------|---------------------------------------------|---------|---------|---------|------------|-------------|
| Aggregation        | Best                                        |         |         |         | Mean           |  Geom_Median           |
| N° Refs            | 1         | 8       | 16      | 64      | All        | All         |
| Size of Word (B)   | 1024     | 8192    | 16384   | 65536   | 1024       | 1024        |
| Quant (B)          | 128      | 1024    | 2048    | 8192    | 128        | 128         |
| Words Float        | 64       | 8       | 4       | 1       | 64         | 64          |
| Words 4-int        | 512      | 64      | 32      | 8       | 512        | 512         |

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

Tested using a balanced dataset of “Sheila” utterances from both Google Speech Commands and custom recordings. Training was done with Edge Impulse using 40×40 Mel-spectrograms, targeting 2 classes: "sheila" and "not sheila". "not sheila" consisted in unknown words, similar words and sounds. 

### SV (Speaker Verification) with CNN Teacher Model

- **Recall**: up to 94% with 64 references (Best aggregation), but too many references, so good trade-off in 
- **F1 Score**: in general good F1 with high number of references
- **Cosine Similarity Matching**: Stable across batches
- **Weakness**: Non-deployable on NDP101, used as reference for distillation
![16 EER](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/2c3e31d64a4494648086e1055daf7e0827ca9b3f/images/5.03%20F1%20Score%2016%20CNN.png)
![16 PREC](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/2c3e31d64a4494648086e1055daf7e0827ca9b3f/images/5.04%20F1%20Score%2016%20CNN%20prec1.png)

### SV with Distilled DNN Models

- **Best deployable configuration**: 256-192 (Intermediate layer = 192 neurons), F1 score around 70% with precision=1
- **Memory usage**: Fits within NDP101 flash constraints
- **F1 Score**: ~0.58 with Best aggregation and 8 references
- **Recall**: ~60% at 100% precision
- **Best memory-efficient configuration**: 128-128 with Mean aggregation, but at the same time too low recall
- **False Positive Rate**: Tunable via threshold; precision=1 achieved with reduced recall
![16 EER](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/45f505e947047a611e86193e648939ce41a16c92/images/5.05%20F1%20Score%2016%20DNN.png)
![16 PREC](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/45f505e947047a611e86193e648939ce41a16c92/images/5.06%20F1%20Score%2016%20DNN%20prec1.png)
![64 EER](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/45f505e947047a611e86193e648939ce41a16c92/images/5.07%20F1%20Score%2064%20DNN.png)
![64 PREC](https://github.com/Gotta003/Syntiant-NDP101-Speaker-Verification-Thesis/blob/45f505e947047a611e86193e648939ce41a16c92/images/5.08%20F1%20Score%2064%20DNN%20prec1.png)
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
- Deploy with SDK if accessible (there are already in deploy folder a working KWS deployable on Syntiant NDP101 and a setup for SV but it is missing 
- Trying optimize these algorithms in DNN
- Port to boards with CNN support to have higher control
- Try changing to more complex and solid Neural Network

---

## Main References

- Edge Impulse
- Google Speech Commands
- TinySV
- LibriSpeech
- Syntiant Docs
The other minor ones are in the thesis pdf file.

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
