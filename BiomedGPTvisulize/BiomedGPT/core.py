import os, sys
from functools import lru_cache
import torch
import numpy as np
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

sys.path.append("/home/howardliu/work_space/BiomedGPT/BiomedGPT")

from fairseq import tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from utils import checkpoint_utils
from utils.eval_utils import eval_step
# from utils.zero_shot_utils import encode_text
from omegaconf import OmegaConf


def encode_text(task, text):
    bpe = task.bpe.encode(text)
    tokens = task.source_dictionary.encode_line(bpe, add_if_not_exist=False, append_eos=False).long()
    return tokens


def load_image(image_path, image_size=224):
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    image_np = np.array(image).astype(np.float32) / 255.0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    return image_np, transform(image).unsqueeze(0)  # image_np for plotting

@lru_cache
def predict_caption(image_path, prompt, model_path = "/home/howardliu/work_data/iu_xray.pt", bpe_dir="/work104/irischen/BiomedGPT/BiomedGPTvisulize/BiomedGPT/utils/BPE", layer=5, head=0):
    overrides = {"bpe_dir": bpe_dir}
    use_cuda = torch.cuda.is_available()
    use_fp16 = False

    # Load model & task
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], arg_overrides=overrides)
    model = models[0]
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()
    model.prepare_for_inference_(cfg)

    # Register attention hook
    GLOBAL_ATTN = {}

    def hook_fn(module, inp, out):
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            GLOBAL_ATTN['attn'] = out[1].detach().cpu()

    model.encoder.layers[layer].self_attn.register_forward_hook(hook_fn)

    # Prepare input
    image_np, image_tensor = load_image(image_path)
    image_tensor = image_tensor.half().cuda() if use_fp16 else image_tensor.cuda()
    src_tokens = encode_text(task, prompt)
    src_tokens = src_tokens.unsqueeze(0).cuda()
    src_lengths = torch.tensor([len(src_tokens[0])]).cuda()

    sample = {
        "id": torch.tensor([0]),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": image_tensor,
            "patch_masks": torch.tensor([True]).cuda()
        }
    }

    # Predict
    generator = task.build_generator([model], cfg.generation)
    with torch.no_grad():
        result, _ = eval_step(task, generator, [model], sample)
    caption = result[0]["caption"]
    print("✅ Caption:", caption)

    # Attention overlay
    attn_tensor = GLOBAL_ATTN['attn'][head, 0]  # [num_queries, num_keys]
    attn_map = attn_tensor[0, 1:197]  # image patch tokens only
    attn_map = attn_map.reshape(14, 14)
    attn_map = torch.nn.functional.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze().numpy()

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(image_np, cmap='gray')
    ax.imshow(attn_map, cmap='jet', alpha=0.5)
    ax.axis('off')
    os.makedirs("attention_vis", exist_ok=True)
    out_path = f"attention_vis/{os.path.basename(image_path).split('.')[0]}_L{layer}_H{head}.png"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Attention overlay saved to {out_path}")

    return caption, out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--bpe-dir", type=str, required=True, help="Path to BPE directory")
    args = parser.parse_args()

    _prompt = "what does the image describe?"
    predict_caption(args.image_path, args.model_path, args.bpe_dir, _prompt)
