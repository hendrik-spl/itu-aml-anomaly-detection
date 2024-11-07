# itu-aml-anomaly-detection

## Environment creation

### Set up venv environment
```bash
python -m venv .venv
pip install -r pip_requirements.txt
```

### Activate your venv environment
```bash
source .venv/bin/activate
```

### Set up environment variables
* duplicate the '.env.sample' file, rename to '.env' and the relevant credentials. The credentials can be found on Google Drive. 

### Set up Weights and Biases
* run 'wandb login' to initialize your login and add your API key when prompted