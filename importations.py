import io
import torch
from PIL import Image
import numpy as np
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from ip_adapter import IPAdapter
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import sys
sys.path.append('/content/BrainACTIV')
from BrainACTIV.dataset.nsd_clip import CLIPExtractor
from BrainACTIV.methods.dino_encoder import EncoderModule, DINO_TRANSFORM
from BrainACTIV.methods.slerp import slerp