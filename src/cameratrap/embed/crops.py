"""Cut the crop every arm sees: the most confident MDv5a animal box.

Boxes are enlarged a little and squared before cropping. Backbones resize
the short side and centre-crop, so a long thin box (a walking tapir, a
guan's tail) would otherwise lose its ends.
"""

from PIL import Image

# Fixed before the spike (provisional judgement, 2026-10-28).
PAD = 0.1


def top_animal(detections, min_conf=0.0):
    """Return the most confident animal detection, or ``None``.

    Args:
        detections: MDv5a detections for one image.
        min_conf: Minimum confidence.

    Returns:
        The detection dict, or ``None`` if no animal box reaches ``min_conf``.
    """
    animals = [d for d in detections if d["category"] == "1" and d["conf"] >= min_conf]
    return max(animals, key=lambda d: d["conf"], default=None)


def square_box(bbox, width, height, pad=PAD):
    """Turn a normalised box into a padded, square pixel box inside the image.

    Args:
        bbox: ``[x, y, w, h]`` normalised to the image size, top-left origin.
        width: Image width in pixels.
        height: Image height in pixels.
        pad: Fraction added to each side of the box before squaring.

    Returns:
        ``(left, top, right, bottom)`` integer pixel coordinates. The square
        is shifted to stay inside the image and only shrinks if it is
        larger than the image's short side.
    """
    x, y, w, h = bbox
    cx, cy = (x + w / 2) * width, (y + h / 2) * height
    side = max(w * width, h * height) * (1 + 2 * pad)
    side = min(side, width, height)
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return round(left), round(top), round(left + side), round(top + side)


def crop(path, detection):
    """Open an image and cut its square crop around a detection.

    Args:
        path: Path to the image file.
        detection: The detection to crop, with a normalised ``bbox``.

    Returns:
        The crop as an RGB PIL image.
    """
    with Image.open(path) as image:
        image = image.convert("RGB")
        return image.crop(square_box(detection["bbox"], *image.size))
