import cv2
import numpy as np

def debayer_2x2(bayer_img):
    """
    Full-resolution 2x2 debayer (strict).
    Uses only the 2x2 Bayer cell (no 3x3 interpolation).

    Assumes RGGB:
        R G
        G B

    Output: same H×W, BGR
    """
    h, w = bayer_img.shape

    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("Image dimensions must be even")

    b = bayer_img.astype(np.uint16)

    # Extract 2x2 blocks
    R  = b[0::2, 0::2]
    G1 = b[0::2, 1::2]
    G2 = b[1::2, 0::2]
    B  = b[1::2, 1::2]

    # Average green
    G = (G1 + G2) // 2

    # Allocate full-resolution output
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Fill each position using ONLY the 2x2 block values

    # Top-left (R location)
    out[0::2, 0::2, 2] = R.astype(np.uint8)   # R
    out[0::2, 0::2, 1] = G.astype(np.uint8)   # G
    out[0::2, 0::2, 0] = B.astype(np.uint8)   # B

    # Top-right (G location)
    out[0::2, 1::2, 2] = R.astype(np.uint8)
    out[0::2, 1::2, 1] = G.astype(np.uint8)
    out[0::2, 1::2, 0] = B.astype(np.uint8)

    # Bottom-left (G location)
    out[1::2, 0::2, 2] = R.astype(np.uint8)
    out[1::2, 0::2, 1] = G.astype(np.uint8)
    out[1::2, 0::2, 0] = B.astype(np.uint8)

    # Bottom-right (B location)
    out[1::2, 1::2, 2] = R.astype(np.uint8)
    out[1::2, 1::2, 1] = G.astype(np.uint8)
    out[1::2, 1::2, 0] = B.astype(np.uint8)

    return out
def demosaic_bayer(raw_path, output_path, pattern="RGGB"):
    """
    Convert a Bayer RAW image (single channel) back to a color image.
    """

    # Load RAW as single channel (IMPORTANT)
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError("Could not load RAW image")

    # Map pattern to OpenCV conversion code
    pattern_map = {
        "RGGB": cv2.COLOR_BayerRG2BGR,
        "BGGR": cv2.COLOR_BayerBG2BGR,
        "GRBG": cv2.COLOR_BayerGR2BGR,
        "GBRG": cv2.COLOR_BayerGB2BGR,
    }

    if pattern not in pattern_map:
        raise ValueError("Invalid Bayer pattern")

    # Demosaic
    color = cv2.cvtColor(raw, pattern_map[pattern])
    
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    # Save result
    cv2.imwrite(output_path, color)
    print(f"Saved demosaiced image to {output_path}")


def main():
    raw_path = "galaxy_raw.png"
    output_path = "reconstructed.png"

    # Must match what you used earlier!
    bayer_pattern = "RGGB"

    demosaic_bayer(raw_path, output_path, pattern=bayer_pattern)
    
    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    result = debayer_2x2(raw)
    cv2.imwrite('debayered_2x2.png', result)


if __name__ == "__main__":
    main()
