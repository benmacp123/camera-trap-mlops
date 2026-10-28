"""Tests for crop geometry: crops are padded squares that stay inside the
image, cut around the most confident animal box.
"""

from PIL import Image

from cameratrap.embed import crops


def test_square_box_is_padded_square():
    """A wide box becomes a square on its long side plus padding."""
    left, top, right, bottom = crops.square_box(
        [0.4, 0.4, 0.2, 0.1], 1000, 1000, pad=0.1
    )
    assert right - left == bottom - top == 240
    assert (left + right) / 2 == 500 and (top + bottom) / 2 == 450


def test_square_box_shifts_inside_image():
    """A box at the edge is shifted, not cut, so the crop stays square."""
    box = crops.square_box([0.0, 0.9, 0.1, 0.1], 1000, 800, pad=0.0)
    assert box == (0, 700, 100, 800)


def test_square_box_never_exceeds_short_side():
    """A box larger than the image shrinks to the image's short side."""
    left, top, right, bottom = crops.square_box([0.0, 0.0, 1.0, 1.0], 1920, 1440)
    assert (right - left, bottom - top) == (1440, 1440)
    assert 0 <= left and right <= 1920


def test_top_animal_ignores_people_and_threshold():
    """The crop is the most confident animal box at or above the threshold."""
    dets = [
        {"category": "2", "conf": 0.99},
        {"category": "1", "conf": 0.3},
        {"category": "1", "conf": 0.6},
    ]
    assert crops.top_animal(dets)["conf"] == 0.6
    assert crops.top_animal(dets, min_conf=0.7) is None


def test_crop_cuts_the_box(tmp_path):
    """The crop has the square box's size and is RGB."""
    path = tmp_path / "x.jpg"
    Image.new("L", (400, 300)).save(path)
    image = crops.crop(path, {"bbox": [0.25, 0.25, 0.25, 0.25]})
    assert image.mode == "RGB"
    assert image.size == (120, 120)
