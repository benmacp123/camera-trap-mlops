"""The promoted model end to end: crop files in, class logits out.

One core for both routes. The local runner and the Lambda handler each
call ``load`` once and ``predict`` per batch, so the cloud route cannot
drift from the model the gate promoted: DINOv2 ViT-B/14 at 224 px with
the linear head fitted on the 39 fold sites.
"""

import dataclasses

from cameratrap.arms import head as head_module
from cameratrap.embed import dinov2


@dataclasses.dataclass
class Champion:
    """A loaded champion.

    Attributes:
        model: The DINOv2 encoder, in eval mode.
        preprocess: Transform from a PIL image to the encoder's input.
        head: The linear head (``coef``, ``intercept``, ``classes``).
    """

    model: object
    preprocess: object
    head: dict


def load(weights=None, head_path=head_module.CHAMPION_HEAD):
    """Load the encoder and the promoted head.

    Args:
        weights: Path to a local DINOv2 ``model.safetensors``; ``None``
            uses the Hugging Face cache.
        head_path: The saved head.

    Returns:
        A ``Champion``.
    """
    model, preprocess = dinov2.create(weights)
    return Champion(model, preprocess, head_module.load(head_path))


def predict(champion, files, batch_size=32):
    """Return the champion's logits for some crop files.

    Args:
        champion: A ``Champion`` from ``load``.
        files: Paths of the crops, in output order.
        batch_size: Crops per forward pass.

    Returns:
        A float64 array of shape ``(len(files), n_classes)``, columns in
        ``champion.head["classes"]`` order. Sequence-level predictions come
        from averaging these across a sequence (``arms.scoring``).
    """
    vectors = dinov2.embed_files(champion.model, champion.preprocess, files, batch_size)
    return head_module.logits(champion.head, vectors)
