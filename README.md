# Real-Time Bit Performance Monitoring: A Dual-Dimensional Graph Attention Network


This repository contains the implementation of a real-time bit performance monitoring framework based on a **Dual-Dimensional Graph Attention Network (DD-GAT)** with multivariate time-series drilling data.



## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
Zhang, Rui and Zhu, Zhaopeng and Ye, Shanlin and Song, Xianzhi and Wu, Yi and Li, Bingxuan and Liu, Haotian and Bijeljic, Branko and Blunt, Martin J. "Real-Time Bit Performance Monitoring: A Dual-Dimensional Graph Attention Network with Multivariate Time-Series Data." Paper presented at the SPE Annual Technical Conference and Exhibition, Houston, TX, USA, October 2025.
```



## 🎯 Overview

Efficient bit performance is crucial for optimizing drilling operations. This framework addresses the limitations of traditional empirical parameters and data-driven models by providing:

- **Real-time monitoring** of drilling bit performance using surface sensor data

- **High accuracy detection** (>95%) of bit anomalies including severe wear, stick-slip vibrations, and bit balling

- **Interpretable results** through dual-dimensional attention mechanisms

- **Cost-effective solution** reducing dependence on expensive downhole measurement tools

  

## 🏗️ Architecture

The framework consists of three core modules:

### 1. Data Preprocessing

- Feature augmentation using expert drilling knowledge
- Construction of drilling condition-related feature sets
- Data cleaning and normalization

### 2. Real-Time Monitoring  

- **Feature-Temporal Dual-Dimensional Graph Attention Network**
- Captures inter-relationships among drilling parameters
- Models temporal dependencies in drilling sequences
- Joint optimization of prediction-based and reconstruction-based models

### 3. Analysis and Interpretation

- Two-dimensional attention mechanism for interpretability
- Anomaly tracing and analysis capabilities
- Engineering decision support

## 🚀 Key Features

- ✅ **Self-supervised learning** framework reducing reliance on labeled data

- ✅ **Dual-dimensional attention** capturing both feature interactions and temporal patterns

- ✅ **Real-time processing** suitable for live drilling operations

- ✅ **Interpretable outputs** supporting engineering decisions

- ✅ **Robust performance** in noisy downhole environments

- ✅ **Scalable architecture** for different drilling scenarios

  

## 📦 Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.1.0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## 



## Acknowledgments

- China University of Petroleum, Beijing
- Imperial College London
- CNOOC Research Institute Co., Ltd
- National Key Laboratory of Petroleum Resources and Engineering
