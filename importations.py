import io
import torch
from PIL import Image
import numpy as np
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from ip_adapter import IPAdapter

from dataset.nsd_clip import CLIPExtractor
from methods.dino_encoder import EncoderModule, DINO_TRANSFORM
from methods.slerp import slerp