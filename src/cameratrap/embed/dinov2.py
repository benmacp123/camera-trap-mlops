"""DINOv2 ViT-B/14 at 224 px: the champion's image encoder.

Kept apart from ``backbones`` so the Lambda container can import it with
torch, timm and Pillow alone: no open_clip, polars or Hugging Face
download. ``backbones.load_dinov2`` builds its model here too, so the
local cache and the cloud route share one definition of the model and its
preprocessing.
"""

import numpy as np
import timm
import torch
from PIL import Image

MODEL_ID = "timm/vit_base_patch14_dinov2.lvd142m"
# DINOv2's default 518 px runs at 1 crop/s on CPU; 224 px runs at 7.5 and
# suits crops whose long side is capped at 448. Fixed before Phase 2.
SIZE = 224


def create(weights=None):
    """Build the encoder and its preprocessing.

    Args:
        weights: Path to a local ``model.safetensors``; ``None`` loads the
            weights through the Hugging Face cache, downloading if needed.

    Returns:
        A pair ``(model, preprocess)``: the model in eval mode with its
        pooled (CLS) output, and the transform from a PIL image to its
        input tensor.
    """
    # A local file replaces only where the weights come from; the rest of
    # the pretrained config (mean, std, interpolation) stays timm's.
    overlay = {"file": str(weights)} if weights else None
    model = timm.create_model(
        MODEL_ID.removeprefix("timm/"),
        pretrained=True,
        pretrained_cfg_overlay=overlay,
        num_classes=0,
        img_size=SIZE,
    ).eval()
    # The pretrained config still says 518 px after img_size=224, so the
    # input size is passed explicitly or the transform would resize to 518.
    config = timm.data.resolve_data_config({"input_size": (3, SIZE, SIZE)}, model=model)
    return model, timm.data.create_transform(**config)


def embed_files(model, preprocess, files, batch_size=32):
    """Return L2-normalised embeddings for image files, in one process.

    No DataLoader workers: Lambda has no shared memory for them, and a
    shard is small enough to decode in the main process.

    Args:
        model: The encoder from ``create``.
        preprocess: Its transform.
        files: Paths of the images, in output order.
        batch_size: Images per forward pass.

    Returns:
        A float32 array with one row per file.
    """
    rows = []
    with torch.no_grad():
        for start in range(0, len(files), batch_size):
            tensors = []
            for path in files[start : start + batch_size]:
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            features = model(torch.stack(tensors))
            rows.append(torch.nn.functional.normalize(features, dim=-1).numpy())
    if not rows:
        return np.zeros((0, model.num_features), dtype=np.float32)
    return np.concatenate(rows).astype(np.float32)


def threads():
    """Return how many CPU threads torch will use for a forward pass.

    Returns:
        The intra-op thread count.
    """
    return torch.get_num_threads()
