# Deepfake Audio Detection: Pied Piper

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-red.svg)](https://github.com/ub-cse676-a-s26/final-project-pied-piper)

A comprehensive deep learning pipeline for detecting synthesized (AI-generated) speech in real-world acoustic environments.

This project progresses through three stages:

- **Baseline CNN:** A lightweight convolutional model trained and evaluated on individual datasets to establish initial performance benchmarks and understand dataset-specific characteristics  
- **Phase 1:** Establishes a domain-balanced ResNet-18 baseline trained on combined datasets for improved generalization  
- **Phase 2:** Leverages self-supervised learning (**Wav2Vec 2.0**) to achieve robust cross-condition generalization   

---

## 📊 Final Results

- **EnvSDD (In-distribution):** Macro F1 = **0.9948** - **LENS-DF (Cross-domain):** - Phase 1: **0.6880** - Phase 2: **0.8592**

---

## 📁 Project Structure

```text
Deepfake-Audio-Detection/
├── app.py                                     # Streamlit web application
├── predict.py                                 # Inference script for local audio files
├── train_baseline.py                          # Phase 1 baseline training script
├── split_dataset.py                           # Data preprocessing & stratified splitting
├── lens_df.py                                 # LENS-DF dataset loader
├── analyze_predictions.py                     # Post-hoc analysis & visualization
│
├── models/                                    # Trained model checkpoints
│   ├── best_model.pth                         # General checkpoint
│   ├── best_model_phase1_rverma2.pth          # Phase 1 ResNet-18 model
│   ├── best_model_wav2vec.pth                 # Phase 2 Wav2Vec 2.0 model
│   └── best_model_2604.pth                    # Final deployment model
│
├── notebooks/
│   ├── Phase_1_CombinedDataset_&_training_rverma2.ipynb
│   ├── Final_Project_latest_Wave2Vec.ipynb
│   └── Final_Project_latest2604.ipynb
│
├── data/
│   ├── LA_bonafide_10_0.wav
│   ├── LA_spoof_3_7_980.wav
│   ├── LA_spoof_3_7_999.wav
│   └── Lens_Bonafide/
│
├── results/
│   ├── Phase_1_CombinedDataset_&_training_results_summary_rverma2.csv
│   ├── rand_test_predictions.csv
│   └── rand_test_summary.csv
│
├── documentation/
│   ├── Deep_Learning_Final_Project_Poster.pdf
│   ├── Deep_Learning_Final_Project_Poster.pptx
│   ├── DemoDayPoster/
│   └── rajagop6_rverma2_jadenpea.pdf
│
├── figures/
│   └── Figure_1.png
│
├── scripts/
│   └── Lens-Df.py
│
├── models_artifacts/
│   └── drive-download-20260429T004500Z-3-001
│
├── README.md
└── .gitattributes
```

📦 Datasets
The datasets used in this project can be accessed here:

https://zenodo.org/records/15241138

https://zenodo.org/records/15220951

https://zenodo.org/records/15948624


🚀 Key Highlights
Domain-balanced training significantly improves generalization

Wav2Vec 2.0 enables strong cross-dataset performance gains

End-to-end pipeline: preprocessing → training → evaluation → deployment

Streamlit app for real-time inference
