# BrainACTIV-SLURM

## Set up
### Set up Virtual environment
```
python -m venv venv
.\venv\Scripts\activate
```

### Install requirements
```
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Download
```
python -m gdown https://drive.google.com/uc?id=1Wi4V2HFss6omhg557FgbRiEBSFxJCPXZ
unzip /brainACTIV_subj1_checkpoints.zip -d /downloads/checkpoints/
rm /brainACTIV_subj1_checkpoints.zip
```