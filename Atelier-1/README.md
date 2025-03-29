# PyTorch-Based Predictive Maintenance and Stock Prediction

This repository contains code implementing deep neural network (DNN) models using PyTorch for two distinct tasks:

*   **Predictive Maintenance (Multi-class Classification):** Predicting machine failure types based on operational data.
*   **Stock Prediction (Regression):** Predicting closing stock prices using historical stock data.

This project was undertaken as part of the Deep Learning Module for a master's level course in a school of technology. The objectives of this atelier/lab were to familiarize oneself with the PyTorch library and DNN/MLP architectures.

## Repository Structure

```
├── Part1_Regression/
| ├── reg.ipynb # Regression notebook
| └── prices.csv # Regression Dataset
├── Part2_Classification/
| ├── classif.ipynb # Classification notebook
| └── predictive_maintenance.csv # Multi-class Classification Dataset
└── README.md
```


## Datasets

**Important:** Due to size limitations, the datasets (`predictive_maintenance.csv` and `prices.csv`) are **not included** directly in the repository. You will need to download them separately and place them alongside their corresponding notebook.

*   **Predictive Maintenance Dataset:** Available at [https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) Place this file in the `Part2_Classification/` directory.
*   **Stock Prediction Dataset:** Available at [https://www.kaggle.com/datasets/dgawlik/nyse](https://www.kaggle.com/datasets/dgawlik/nyse) Place this file in the `Part1_Regression/` directory.

Your `DL--Atelier1` directory should look like this:

```
DL--Atelier1/
├── Part1_Regression/
| ├── reg.ipynb
| └── prices.csv
├── Part2_Classification/
| ├── classif.ipynb
| └── predictive_maintenance.csv
└── README.md
```


## Overview

This repository demonstrates end-to-end deep learning workflows, from data exploration and preprocessing to model building, training, and evaluation. The notebooks provide a clear, step-by-step guide, suitable for both learning and practical application.

## Project Breakdown

### 1. Stock Prediction (Regression) - `Part1_Regression/reg.ipynb`

*   **Objective:** Predict closing stock prices.
*   **Key Steps:**
    *   **Data Loading and Exploration:** Loading the stock dataset using pandas and performing exploratory data analysis (EDA) to get an overview of the data.
    *   **Data Preprocessing:** Cleaning and preparing the data for the model.
    *   **Model Building:** Creating a DNN model using PyTorch.
    *   **Training:** Training the model on the prepared data.
*   **Techniques Used:**
    *   `pandas`: Data loading, exploration, and manipulation.
    *   `matplotlib`, `seaborn`: Data visualization.
    *   `torch`, `torch.nn`, `torch.optim`: Deep Learning and model building.
    *   `sklearn.preprocessing`: Data scaling.
    *   `torch.utils.data`: Creating Dataset and DataLoaders.

*   **Colab Notebook:** [https://colab.research.google.com/drive/1Uhvs37ZMI57GPRskYAb_LH5qptSPZLQn?usp=sharing](https://colab.research.google.com/drive/1Uhvs37ZMI57GPRskYAb_LH5qptSPZLQn?usp=sharing)

### 2. Predictive Maintenance (Multi-class Classification) - `Part2_Classification/classif.ipynb`

*   **Objective:** Predict machine failure types.
*   **Key Steps:**
    *   **Data Loading and Exploration:** Loads the data and uses EDA for understanding and visualizing the dataset.
    *   **Data Cleaning and Preprocessing:** Addresses missing values, duplicates, and encodes categorical variables using one-hot encoding.
    *   **Data Augmentation:** Implements SMOTE to handle imbalanced class distributions.
    *   **Model Building:** Implements DNN models using PyTorch.
    *   **Grid Search:** Uses a grid search approach to find the optimal hyperparameter configuration for the model.
    *   **Evaluation:** Computes performance metrics (Accuracy, Precision, Recall, and F1-score) and analyzes the Confusion matrix.
    *   **Regularization:** Adds L2-Regularization and Dropout to prevent overfitting.
*   **Techniques Used:**
    *   `pandas`: Data loading, exploration, and manipulation.
    *   `matplotlib`, `seaborn`: Data visualization.
    *   `torch`, `torch.nn`, `torch.optim`: Deep Learning and model building.
    *   `sklearn.preprocessing`: Data scaling and encoding.
    *   `torch.utils.data`: Creating Dataset and DataLoaders.
    *   `imblearn.over_sampling`: Balancing the dataset using SMOTE.
*   **Colab Notebook:** [https://colab.research.google.com/drive/11Sso07vkii6dyZSW2ZezkRjvNFx_8G4m?usp=sharing](https://colab.research.google.com/drive/11Sso07vkii6dyZSW2ZezkRjvNFx_8G4m?usp=sharing)

## Running the Code

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Medamine-Bahassou/DL--Atelier1
    cd DL--Atelier1
    ```

2.  **Download the datasets:** Download `predictive_maintenance.csv` and `prices.csv` from Kaggle. Then, place them in the following locations:
    *   `predictive_maintenance.csv`  -> `Part2_Classification/`
    *   `prices.csv` -> `Part1_Regression/`

3.  **Install the required libraries:**

    ```bash
    pip install pandas scikit-learn torch matplotlib seaborn imbalanced-learn
    ```

4.  **Open and run the notebooks:** Open `Part1_Regression/reg.ipynb` and `Part2_Classification/classif.ipynb` in Jupyter Notebook or Google Colab and execute the code cells sequentially.

## Key Findings

*   **Regression:** The PyTorch DNN model showed that Stock prediction is possible with an acceptable MSE, however, it still needs tuning to prevent overfitting and improve generalization. More extensive hyperparameter optimization and feature engineering could further enhance performance.
*   **Classification:** The DNN model achieved high accuracy on the test set with SMOTE balancing the dataset, demonstrating effective classification of machine failure types. The importance of hyperparameters and regularization was also demonstrated, as the initial model without these techniques performed significantly worse.

## Credit

This project is an atelier/lab assignment for a Deep Learning module.
