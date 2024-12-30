
# Automated Essay Scoring (AES) Project

![image](https://github.com/user-attachments/assets/82e2e1a8-bd23-48a6-acae-f96b74bbf683)


## Overview

This repository contains all the notebooks, resources, and documentation used to develop and evaluate models for the **Automated Essay Scoring (AES)** Kaggle competition. The project aims to build an open-source solution for automated essay evaluation to support educators and provide timely feedback to students.

- [Competition link](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/overview)
- [Training Notebooks](https://github.com/Jatin-Mehra119/Essay-Scoring-Modeling/tree/main/Research%20Notebooks)
- [Final Model](https://huggingface.co/jatinmehra/Smollm2-360M-Essay-Scoring)
- [Essay Scorer Pro APP](https://huggingface.co/spaces/jatinmehra/Essay-Scorer-Pro)
----------

## Project Objectives

1.  **Develop Accurate Scoring Models**  
    Create models capable of predicting essay scores with high agreement to human evaluators using the **Quadratic Weighted Kappa (QWK)** metric.
    
2.  **Support Educators**  
    Reduce manual grading workload and provide consistent scoring across diverse writing samples.
    
3.  **Ensure Fairness**  
    Mitigate algorithmic bias across different demographics and writing styles.
    
4.  **Enhance Feedback Loop**  
    Offer timely, detailed, and constructive feedback to students for continuous improvement.
    
5.  **Resource Efficiency**  
    Design scalable and efficient models for deployment in diverse educational environments.
    

----------

## Dataset

The dataset, provided by the competition host, includes essays scored on a 1–6 scale. The training data consists of rich textual features aligned with classroom standards, ensuring diversity and fairness.

----------

## Evaluation Metric

The **Quadratic Weighted Kappa (QWK)** metric is used to evaluate model performance.

-   **QWK Range**: -1 (worse than random) to 1 (perfect agreement).
-   The metric penalizes large deviations between predicted and actual scores.

----------

### **Inference Performance**

-   **Batch Size:** 16
-   **Number of Samples:** 17,000
-   **Device:** NVIDIA Tesla P100 GPU |  Intel(R) Xeon(R) CPU @ 2.20GHz
-   **Average Inference Time per Sample:** ~56ms
-   **Total Inference Time:** ~16 minutes for 17,000 samples

----------

## Project Roadmap

### 1. Data Preparation

-   Resolved encoding issues and normalized text.
-   Engineered features like text length, spelling mistakes, and stopword ratios.

### 2. Exploratory Data Analysis (EDA)

-   Analyzed score distributions and textual patterns.
-   Visualized relationships between features and scores.

### 3. Baseline Models

-   Linear regression achieved **63 QWK**.

### 4. Advanced Models

-   **LightGBM**:
    -   Base: **70 QWK**
    -   Hypertuned: **74 QWK**
-   **Fine-Tuned Smollm2 (360M)**:
    -   Achieved **79 QWK**.

### 5. Visualization

-   Word clouds and feature histograms for different score levels.

### 6. Deployment

-   Fine-tuned Smollm2 model is available on Hugging Face Hub: [Smollm2-360M-Essay-Scoring](https://huggingface.co/jatinmehra/Smollm2-360M-Essay-Scoring).

----------

## Models

### 1. Baseline Models

-   **Linear Regression**: Established initial benchmarks.

### 2. LightGBM

-   Tuned hyperparameters using Optuna.
-   Utilized weighted loss to address class imbalance.

### 3. Smollm2 (360M)

-   Fine-tuned a transformer-based model for essay scoring.
-   Achieved the best performance with **79 QWK**.

----------

## Results


| Model                        | QWK Score |
|------------------------------|-----------|
| Baseline Linear Regression    | 63        |
| LightGBM (Base)               | 70        |
| LightGBM (Hypertuned)         | 74        |
| Smollm2 (Fine-Tuned)         | 79        |

----------

## Hugging Face Model



The fine-tuned model is hosted on Hugging Face:  
[Smollm2-360M-Essay-Scoring](https://huggingface.co/jatinmehra/Smollm2-360M-Essay-Scoring) (Includes Guide how to use this model)


## Future Work

-   Explore additional datasets for improved generalization.
-   Develop a user-friendly interface for educators and students.
-   Experiment with larger transformer models for better performance.

----------

## Acknowledgments

This project is supported by the Kaggle competition dataset and resources. Special thanks to the Hugging Face team for their robust model and library support.

-----------
## Citation

```
@misc {jatin_mehra_2024,
    author       = { {Jatin Mehra} },
    title        = { Smollm2-360M-Essay-Scoring (Revision 467ceb5) },
    year         = 2024,
    url          = { https://huggingface.co/jatinmehra/Smollm2-360M-Essay-Scoring },
    doi          = { 10.57967/hf/3924 },
    publisher    = { Hugging Face }
}

@misc{learning-agency-lab-automated-essay-scoring-2,
    author = {Scott Crossley and Perpetual Baffour and Jules King and Lauryn Burleigh and Walter Reade and Maggie Demkin},
    title = {Learning Agency Lab - Automated Essay Scoring 2.0},
    year = {2024},
    howpublished = {\url{https://kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2}},
    note = {Kaggle}
}

@misc{allal2024SmolLM2,
      title={SmolLM2 - with great data, comes great performance}, 
      author={Loubna Ben Allal and Anton Lozhkov and Elie Bakouch and Gabriel Martín Blázquez and Lewis Tunstall and Agustín Piqueres and Andres Marafioti and Cyril Zakka and Leandro von Werra and Thomas Wolf},
      year={2024},
}

```
