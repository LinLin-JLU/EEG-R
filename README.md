# EEG Emotion Recognition using Spatial-Temporal Graph-Aware Network (STG-Net) with Channel Selection

This is a PyTorch implementation of the paper: "EEG-Based Emotion Recognition Using Spatial-Temporal Graph-Aware Network with Channel Selection".

This framework introduces a novel model (STG-Net) for EEG-based emotion recognition. It first adaptively selects the most informative EEG channels by combining **Wavelet Coherence** and **Mutual Information**. It then uses a hierarchical spatial-temporal model to extract and fuse features, enhancing both performance and efficiency.

The model was validated on the SEED and DEAP datasets

## 🧠 Model Architecture (STG-Net)

The core architecture of STG-Net is divided into three main parts:

1.  **(a) Channel Selection**:
    * Uses **Wavelet Coherence** to calculate inter-channel correlations to identify redundancy.
    * Uses **Mutual Information** to evaluate the relevance of each channel to the emotion labels.
    * Filters channels using a dynamic threshold ($T=\overline{C}+k\cdot\sigma_{C}$) to reduce redundancy while preserving key information.

2.  **(b) Spatial Information Processing**:
    * Extracts **Differential Entropy (DE)** features from the selected channels as the initial node features for the graph.
    * Uses a **two-layer Graph Convolutional Network (GCN)** to capture spatial topological relationships within each time frame.

3.  **(c) Temporal Information Processing**:
    * **Transformer Encoder**: The output from the GCN is fed into a Transformer to capture long-range global temporal dependencies.
    * **LSTM Network**: The Transformer's output is then passed to an LSTM to model the sequential dynamics of emotional states.
    * **Classification**: Finally, the last hidden state ($h_{last}$) from the LSTM is used for classification via a Fully Connected layer and Softmax.

## 🛠️ Requirements

```bash
pip install torch
pip install numpy
pip install pandas
pip install scikit-learn
pip install einops
pip install pywavelets
