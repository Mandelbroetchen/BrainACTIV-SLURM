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
wget https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin
gdown https://drive.google.com/uc?id=1Wi4V2HFss6omhg557FgbRiEBSFxJCPXZ
unzip /content/brainACTIV_subj1_checkpoints.zip -d /content/checkpoints/ && rm /content/brainACTIV_subj1_checkpoints.zip
gdown --folder https://drive.google.com/drive/folders/1_4rNJEhdklBkOt-JeNjrvOdSZaHmF-UE
```