# Non-Intrusive_PESQ-DNN

# PESQ-DNN

This is the inference script referring to the paper "Coded Speech Quality Measurement by a Non-Intrusive PESQ-DNN". In this repository, we provide the source code for testing the trained PESQ-DNN to non-intrusivelly esitmate the PESQ scores of the input coded speech utterances.

The code was written by Ziyi Xu.

## Introduction

Wideband codecs such as AMR-WB or EVS are widely used in (mobile) speech communication. Evaluation of coded speech quality is often performed subjectively by an absolute category rating (ACR) listening test. However, the ACR test is impractical for online monitoring of speech communication networks. Perceptual evaluation of speech quality (PESQ) is one of the widely used metrics instrumentally predicting the results of an ACR test. However, the PESQ algorithm requires an original reference signal, which is usually unavailable in network monitoring, thus limiting its applicability. NISQA is a new non-intrusive neural-network-based speech quality measure, focusing on super-wideband speech signals. In this work, however, we aim at predicting the well-known PESQ metric using a non-intrusive PESQ-DNN model. We illustrate the potential of this model by predicting the PESQ scores of wideband-coded speech obtained from AMR-WB or EVS codecs operating at different bitrates in noisy, tandeming, and error-prone transmission conditions.

## Prerequisites

- [Python](https://www.python.org/) 3.6
- CPU or NVIDIA GPU ([CUDA](https://developer.nvidia.com/cuda-toolkit) 9.0 and [CuDNN](https://developer.nvidia.com/cudnn) 7.0.5 for Tensorflow 1.5.0 GPU version).


## Getting Started

### Installation
- Install [Anaconda](https://www.anaconda.com/)
- Install [Python](https://www.python.org/) 3.6
- Install [TensorFlow](https://www.tensorflow.org/) 1.5.0 and [Keras](https://www.tensorflow.org/) 2.1.4
- Some Python packages need to be installed, please see detailed information in the .yaml file (```environment.yaml```)

### Datasets

Note that in this project the clean speech signals are taken from the [NTT wideband speech database](https://www.ntt-at.com/product/multilingual/). If you want to reproduce the exact results, the test need to be done with the same speech data (see details in the paper).

### Inference Preparation
 - To run the inference script, you need:
1. ```mean_for_Complex_Spectrum.mat``` (The mean value is used for normalizing the complex input spectrum of the PESQ-DNN to zero mean. The shape of the data (1 * 512): First 257 and the rest 255 numbers in the vector represent the mean value of the **non-redundant** real and imaginary parts, respectively, with an FFT size of 512)
2. ```std_for_Complex_Spectrum.mat``` (The standard deviation value is used for normalizing the complex input spectrum of the PESQ-DNN to unit variance. The shape of the data (1 * 512): First 257 and the rest 255 numbers in the vector represent the meastandard deviationn value of the **non-redundant** real and imaginary parts, respectively, with an FFT size of 512)
3. Trained Models:
    - ```TRAINED_MODEL.h5``` (Trained PESQ-DNN model in .h5 format)
- All these three files are saved in './Trained_model/'
4. ```audio_processing.py``` (This Python script contains audio pre-processing functions, e.g., windowing, framing, and the following FFT. Detailed settings are introduced in the paper. This Python script is modified and extracted from part of the work from Maximilian Strake)
5. Inference Scripts:
    - ```Inference_PESQ_DNN_FLE.py``` (This Python script predicts the PESQ scores of the coded speech wave (.wav) file with one of the proposed PESQ-DNN variations employing FLE.)
    - ```Inference_PESQ_DNN_BLE_Attention.py``` (This Python script predicts the PESQ scores of the coded speech wave (.wav) file with one of the proposed PESQ-DNN variations employing BLE and attention pooling.)
- Some example speech files (from the [ITU-T test signals](https://www.itu.int/net/itu-t/sigdb/genaudio/AudioForm-g.aspx?val=1000050) of American English) are placed under the directory: `./Test_data/`. To start your own inference, replace these `.wav` files by your own files. More details are in the Python scripts.

### PESQ-DNN Inference

 - Run the Python script to predict the PESQ socres of the speech files stored under the directory: `./Audio_Samples/`:
```bash
python Inference_PESQ_DNN_FLE.py
```
or
```bash
python Inference_PESQ_DNN_BLE_Attention.py
```
 ## Citation

If you use the scripts in your research, please cite

```
@article{xu2023PESQDNN,
  author =  {Z. Xu, Z. Zhao and T. Fingscheidt},
  title =   {{Coded Speech Quality Measurement by a Non-Intrusive PESQ-DNN}},
  journal = {arXiv preprint arXiv: ***},
  year =    {2023},
  month =   {Mar.}
}
```

## Acknowledgements
- The author would like to thank Marvin Sach and Jan Pirklbauer for the advice concerning the construction of these source code in GitHub.
- The author would also like to thank Maximilian Strake for sharing the script for audio pre-processing.
