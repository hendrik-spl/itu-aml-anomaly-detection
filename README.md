# itu-aml-anomaly-detection

## Environment creation

### Initialize the environment with init.sh
* Run the following command in the folder you would like to pull the repository into. This will initialize the environment:
```bash
source init.sh
```
* Note: The `init.sh` script already takes care of setting up and activating the `venv` environment. The following steps are provided as a fallback.
* This setup is specifically designed for the use of CUDA GPUs. Adaptions might be neccessary depending on available compute resources.

### Set up venv environment (Fallback)
```bash
python -m venv .venv
pip install -r pip_requirements.txt
```

### Activate your venv environment (Fallback)
```bash
source .venv/bin/activate
```

* Please continue here if setup with `init.sh` was successfull.

### Set up Weights and Biases
* Run `wandb login` to initialize your login and add your API key when prompted.
* If you want the wandb code to execute without issues, please create a project in wandb called `ablation-study`.

### Example: Run the train_autoencoder.py script
* To run the `train_autoencoder.py` script located in the `scripts` folder, use the following command:
```bash
python scripts/train_autoencoder.py
```
* Some other scripts such as the evaluate_model.py leverages argparser. If you want to use that, paste the wandb id of the run you want to evaluate. 
* Ensure that the respective model `*.keras`is available under `models/checkpoints`