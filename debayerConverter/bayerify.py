import cv2
import numpy as np


def get_bayer_indices(pattern):
    """
    Returns channel indices for a 2x2 Bayer tile.
    OpenCV uses BGR ordering: B=0, G=1, R=2
    """
    patterns = {
        "RGGB": np.array([[2, 1],
                          [1, 0]]),
        "BGGR": np.array([[0, 1],
                          [1, 2]]),
        "GRBG": np.array([[1, 2],
                          [0, 1]]),
        "GBRG": np.array([[1, 0],
                          [2, 1]]),
    }
    return patterns[pattern]


def simulate_bayer_raw(img, pattern="RGGB"):
    """
    Convert a BGR image into a single-channel Bayer RAW image.
    """
    h, w, _ = img.shape
    raw = np.zeros((h, w), dtype=img.dtype)

    bayer = get_bayer_indices(pattern)

    for by in range(2):
        for bx in range(2):
            channel = bayer[by, bx]

            # Select pixels in this Bayer position
            raw[by::2, bx::2] = img[by::2, bx::2, channel]

    return raw


def main():
    input_path = "galaxy.png"
    output_path = "galaxy_raw.png"

    # Choose: "RGGB", "BGGR", "GRBG", "GBRG"
    bayer_pattern = "RGGB"

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not load image")

    raw = simulate_bayer_raw(img, pattern=bayer_pattern)

    cv2.imwrite(output_path, raw)
    print(f"Saved Bayer RAW image to {output_path}")


if __name__ == "__main__":
    main()
