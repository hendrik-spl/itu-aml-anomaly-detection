# Exploring Design Choices in Autoencoder-Based Anomaly Detection

## Overview

This study investigates the design choices for autoencoder-based anomaly detection in industrial applications, using the MVTec Anomaly Detection (MVTec AD) dataset. A comprehensive ablation study examines the impact of architectural components, loss functions, and regularization techniques on anomaly detection performance. This repository is supposed to be used in conjunction with the respective research paper which is available upon request. 

### Initialize the environment with init.sh
* Run the following command in the folder you would like to pull the repository into. This will initialize the environment:
```bash
source init.sh
```
* Note: The `init.sh` script already takes care of setting up and activating the `venv` environment. The following steps are provided as a fallback.
* This setup is specifically designed for the use of CUDA GPUs. Adaptions might be neccessary depending on available compute resources.

#### Fallback: Set up venv environment
```bash
python -m venv .venv
pip install -r pip_requirements.txt
```
```bash
source .venv/bin/activate
```

* Please continue here if setup with `init.sh` was successfull.

### Set up Weights and Biases
* Run `wandb login` to initialize your login and add your API key when prompted.
* If you want the wandb code to execute without issues, please create a project in wandb called `ablation-study`.

### Example: Run the train_model.py script
* This script can be used to train an autoencoder based on the parameters set in the config dictionary.
* To run the `train_model.py` script located in the `scripts` folder, use the following command:
```bash
python scripts/train_model.py
```

### Example: Run the evaluate_model.py script
* This script can be used to evaluate a specific, existing model across multiple evaluation methods. 
* To run the `evaluate_model.py` script located in the `scripts` folder, use the following command:
```bash
python scripts/evaluate_model.py --run_id <wandb_run_id>
```
* Replace `<wandb_run_id>` with the actual wandb run ID you want to evaluate.
* Ensure that the respective model `*.keras` is available under `models/checkpoints`.

### Example: Run the predict_model.py script
* 
* To run the `predict.py` script located in the `scripts` folder, use the following command:
```bash
python scripts/predict.py --run_id <wandb_run_id>
```
* Replace `<wandb_run_id>` with the actual wandb run ID you want to use for prediction.
* Ensure that the respective model `*.keras` is available under `models/checkpoints`.

### Example: Fine-tune a pre-trained model with finetune_model.py - only for PoC purposes, not applied in actual paper
* This script can be used to fine-tune a pre-trained autoencoder model based on the parameters set in the config dictionary. For example, a custom trained model could be further finetuned on a lower learning rate. 
* To run the fine-tuning script, use the following command:
```bash
python scripts/finetune_model.py
```
* Ensure that the pre-trained model `*.keras` is available under `models/checkpoints`.