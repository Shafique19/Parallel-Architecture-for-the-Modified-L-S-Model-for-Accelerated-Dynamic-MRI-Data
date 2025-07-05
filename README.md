# Parallel GPU Architecture for the Modified L+S Model in Cardiac Cine MRI

This repository features a CUDA-accelerated implementation of the **Modified Low-Rank plus Sparse (L+S) Model** for cardiac cine MRI reconstruction. It includes **all CUDA kernels** used in the parallel implementation of the model, along with representative datasets, enabling high-performance computing research in medical image reconstruction.

## 🔧 Project Highlights

- Implements the complete Modified L+S model using **parallel CUDA kernels**.
- Leverages **NVIDIA CUDA** to accelerate MRI frame reconstruction on GPUs.
- Achieves **significant speedup** compared to MATLAB CPU-based execution.
- Provides source code, datasets, and profiling tools for reproducibility and further research.

## 📥 Additional Resources

Complementary datasets and the baseline MATLAB CPU implementation are available at:

🔗 **[Download via Figshare](https://figshare.com/s/b28d3066076439ae680d)**

**This package includes:**
- `.mat` cardiac cine MRI datasets
- MATLAB code for the Modified L+S reconstruction
