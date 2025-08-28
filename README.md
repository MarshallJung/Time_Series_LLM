# Predictive Maintenance Demo with Moirai Time-Series Transformer

This repository contains a Google Colab notebook demonstrating how to fine-tune the Moirai time-series foundation model for a predictive maintenance task. The goal is to predict machinery failure on a multivariate dataset from an oil service pump.

This project serves as a practical, end-to-end guide for adapting a large, pre-trained time-series model to a specific, real-world use case.

## Overview

The notebook covers the entire machine learning lifecycle:
-   **Environment Setup**: Correctly installing `uni2ts` and its dependencies in a Google Colab environment.
-   **Data Loading & Cleaning**: Ingesting raw CSV data and making it robustly numerical.
-   **Feature Engineering**: Applying one-hot encoding for categorical features.
-   **Data Preparation**: Transforming the data into the specific `uni2ts`/`gluonts` format required by the model's training pipeline.
-   **Model Fine-Tuning**: Loading a pre-trained Moirai model from Hugging Face Hub and fine-tuning it on our specific dataset using PyTorch Lightning.
-   **Inference**: Using the fine-tuned model to generate predictions on an unseen test set.
-   **Evaluation & Visualization**: Calculating key performance metrics and plotting the results to visually assess the model's performance.

## Prerequisites

-   A Google Account to use Google Colab.
-   The predictive maintenance dataset, which should be saved as a `.csv` file. For this project, we assume it is named `predictive_maintenance_dataset.csv`.

## Step-by-Step Instructions to Replicate

Follow these steps carefully to run the demonstration successfully.

### 1. Set Up the Google Colab Environment

The most critical part of this process is correctly setting up the Python environment in Google Colab, as its pre-installed libraries can conflict with the ones required by `uni2ts`.

**A. Open Google Colab and Configure Runtime**
1.  Go to [colab.research.google.com](https://colab.research.google.com).
2.  Click `File -> New notebook`.
3.  Click `Runtime -> Change runtime type`.
4.  Select **`T4 GPU`** from the "Hardware accelerator" dropdown and click `Save`. This is crucial for performance.

**B. Run the Setup and Installation Cell**
1.  Create a new code cell in your notebook.
2.  Copy and paste the entire code block below into the cell. This block will:
    -   Uninstall conflicting versions of `torchvision` and `torchaudio`.
    -   Install the specific `torch` version compatible with `uni2ts`.
    -   Install the `uni2ts` library directly from its GitHub repository.

```python
# Step 1: Uninstall potentially conflicting pre-installed packages
%pip uninstall -y torchvision torchaudio
%pip install -q --upgrade pip

# Step 2: Install compatible core libraries (torch, torchvision)
%pip install -q "torch==2.4.1" "torchvision==0.19.1" --index-url https://download.pytorch.org/whl/cu121

# Step 3: Install uni2ts from GitHub
%pip install -q "git+https://github.com/SalesforceAIResearch/uni2ts.git#egg=uni2ts[notebook]"

print("\nInstallation complete. Please RESTART YOUR COLAB RUNTIME before proceeding.")
```
3.  Run this cell. It will take a few minutes to complete.

**C. ⚠️ Restart the Runtime ⚠️**
This is the most important step. After the installation cell finishes, you **must** restart the runtime for the changes to take effect.
-   Go to the menu: `Runtime -> Restart session`.
-   Click `Yes` in the confirmation dialog.

### 2. Run the Data Pipeline and Model Training

After the runtime has restarted, you can proceed with the main logic of the notebook.

**A. Run the Consolidated Data Pipeline Cell**
1.  Create a new code cell.
2.  Copy and paste the entire block from our final "**Section 2 & 3 (Final, v7)**" into this cell. This block handles everything from uploading the data to creating the final `DataLoaders`.
3.  Run the cell. It will prompt you to upload your `predictive_maintenance_dataset.csv` file.

**B. Run the Fine-Tuning Cell**
1.  Create a new code cell.
2.  Copy and paste the code from our final "**Section 4: Model Fine-Tuning**" into this cell.
3.  Run the cell. This will start the fine-tuning process. You will see a progress bar from PyTorch Lightning. This is the longest step and may take 10-20 minutes depending on the Colab GPU allocation. The process will stop automatically when the model's performance on the validation set no longer improves.

### 3. Run Prediction and Visualization

**A. Run the Prediction Cell**
1.  Create a new code cell.
2.  Copy and paste the code from our final "**Revised Section 5: Prediction / Inference (Final, Corrected Method)**" into this cell.
3.  Run the cell. This will load your best fine-tuned model and use it to generate predictions on the test data.

**B. Run the Visualization Cell**
1.  Create a new code cell.
2.  Copy and paste the code from our final "**Revised Section 6: Visualization & Evaluation (Corrected)**" into this cell.
3.  Run the cell. This will calculate the final performance metrics and display the plots comparing the model's predictions to the ground truth.

## Key Learnings & Troubleshooting Guide

This project involved several subtle challenges that are common when working with advanced ML libraries. Here are the key takeaways:

-   **Colab Dependency Management**: Google Colab is a fantastic tool, but its pre-installed environment can be a challenge. The robust solution is to aggressively uninstall conflicting packages (`torchvision`), install specific versions of core libraries (`torch`), and then install your target library (`uni2ts`). A runtime restart is non-negotiable after this process.

-   **Data Preparation is Paramount**: The model expects data in a very specific format. Our key learnings were:
    -   **Use `ListDataset` for Custom Data**: For custom data passed to a PyTorch `DataLoader`, `gluonts.dataset.common.ListDataset` is the correct, subscriptable choice, not `PandasDataset`.
    -   **Data Must Be Purely Numerical**: Ensure all columns passed to the model (target and features) are numeric. We used `pd.to_numeric(errors='coerce')` to enforce this and handle any unexpected string values.
    -   **Metadata is Required**: The `uni2ts` transformation pipeline requires certain metadata keys in the dataset dictionary, such as `"freq"`. Forgetting these will lead to `KeyError` exceptions.

-   **Model API Nuances**:
    -   **Use the Library's Transformation Pipeline**: The most robust way to prepare data is to use the transformation chains provided by the estimator itself (e.g., `estimator.train_transform_map`). This guarantees the data is formatted exactly as the model expects, creating all necessary fields like `observed_mask`, `time_id`, and `sample_id`.
    -   **`FinetuneDataset` and `EvalDataset`**: These are the correct wrapper classes from `uni2ts` for applying transformations during a training or evaluation loop.
    -   **`predictor.predict()` for Inference**: The canonical way to generate forecasts is to use the `predictor.predict()` method, which handles the entire prediction pipeline internally. Avoid using `trainer.predict()` for this workflow, as it can lead to API conflicts.

By following the steps outlined above, you should be able to successfully replicate this predictive maintenance demo.
