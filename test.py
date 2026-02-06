import io
import json
import torch
from PIL import Image
import numpy as np
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from ip_adapter import IPAdapter
import matplotlib.pyplot as plt
import argparse
import sys
sys.path.append('./BrainACTIV')
from BrainACTIV.dataset.nsd_clip import CLIPExtractor
from BrainACTIV.methods.dino_encoder import EncoderModule, DINO_TRANSFORM
from BrainACTIV.methods.slerp import slerp

def main():
    # -------------------------------
    # Parse command line arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="ROI-based image transformation using IP-Adapter.")
    parser.add_argument("config", type=str, help="Path to JSON config file")
    parser.add_argument("input_image", type=str, help="Path to input image")
    parser.add_argument("output_image", type=str, help="Path to save output image")
    args = parser.parse_args()

    # -------------------------------
    # Load config JSON
    # -------------------------------
    with open(args.config, 'r') as f:
        config = json.load(f)

    roi = config.get("roi", "EBA")
    maximize = config.get("maximize", True)
    alpha = config.get("alpha", 0.7)
    gamma = config.get("gamma", 0.6)
    seed = config.get("seed", 42)
    resolution = tuple(config.get("resolution", [512, 512]))

    # -------------------------------
    # Setup device
    # -------------------------------
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # -------------------------------
    # Initialize diffusion pipeline
    # -------------------------------
    diffusion_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16
    ).to(device)
    diffusion_pipeline.scheduler = DDIMScheduler.from_config(diffusion_pipeline.scheduler.config)
    diffusion_pipeline.safety_checker = None

    # -------------------------------
    # Initialize IP-Adapter
    # -------------------------------
    ip_model = IPAdapter(
        diffusion_pipeline, 
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", 
        "ip-adapter_sd15.bin", 
        device
    )

    # -------------------------------
    # Initialize CLIP extractor
    # -------------------------------
    clip_extractor = CLIPExtractor(device)

    # -------------------------------
    # Load input image
    # -------------------------------
    image_ref = Image.open(args.input_image).convert("RGB")
    image_ref = image_ref.resize(resolution)
    image_ref_clip = clip_extractor(image_ref).detach().cpu().numpy()

    # -------------------------------
    # Initialize DINO-ViT Encoder
    # -------------------------------
    ckpt_path_dino = f"./checkpoints/subj1_{roi}.ckpt"
    dino_encoder = EncoderModule.load_from_checkpoint(ckpt_path_dino, strict=False).to(device).eval()

    # -------------------------------
    # Load modulation embedding
    # -------------------------------
    ckpt_path_embed = f"./BrainACTIV_modulation_embeddings/subj1_{roi}_mod_embed_{'max' if maximize else 'min'}.npy"
    mod_embed = np.load(ckpt_path_embed)

    endpoint = mod_embed * np.linalg.norm(image_ref_clip)
    embeds = torch.from_numpy(slerp(image_ref_clip, endpoint, 1, t0=alpha, t1=alpha)).unsqueeze(1).to(device)[0]

    # -------------------------------
    # Generate transformed image
    # -------------------------------
    torch.manual_seed(seed)
    with torch.no_grad():
        image_new = ip_model.generate(
            clip_image_embeds=embeds,
            image=image_ref,
            strength=gamma,
            num_samples=1,
            num_inference_steps=50,
            seed=seed
        )[0]

    # -------------------------------
    # Optionally: compute DINO predictions
    # -------------------------------
    dino_pred_ref = dino_encoder(DINO_TRANSFORM(image_ref).to(device).unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean(-1)
    dino_pred_new = dino_encoder(DINO_TRANSFORM(image_new).to(device).unsqueeze(0)).squeeze(0).detach().cpu().numpy().mean(-1)

    # -------------------------------
    # Save output image
    # -------------------------------
    image_new.save(args.output_image)
    print(f"Transformed image saved to {args.output_image}")

    # -------------------------------
    # Optional: visualize
    # -------------------------------
    f, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(image_ref)
    axes[0].set_title('Reference image')
    axes[0].axis('off')
    axes[1].imshow(image_new)
    axes[1].set_title('Variation')
    axes[1].axis('off')
    axes[2].bar(['Reference', 'Variation'], [dino_pred_ref, dino_pred_new])
    axes[2].set_title(f'Predicted activation for {roi}')
    plt.tight_layout()
    plt.show()

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()
