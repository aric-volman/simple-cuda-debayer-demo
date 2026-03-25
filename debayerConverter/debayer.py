import cv2
import numpy as np

def debayer_2x2(bayer_img):
    """
    Simple 2x2 debayer assuming RGGB pattern:
    
    R G
    G B

    Output is downsampled by 2x (H/2 × W/2), BGR format.
    """
    h, w = bayer_img.shape

    # Ensure even dimensions
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("Image dimensions must be even")

    # Convert to larger type to avoid overflow during averaging
    bayer = bayer_img.astype(np.uint16)

    # Reshape into 2x2 blocks
    reshaped = bayer.reshape(h // 2, 2, w // 2, 2)

    # Extract RGGB components
    # [[R, G],
    #  [G, B]]
    R  = reshaped[:, 0, :, 0]
    G1 = reshaped[:, 0, :, 1]
    G2 = reshaped[:, 1, :, 0]
    B  = reshaped[:, 1, :, 1]

    # Average the two green pixels safely
    G = (G1 + G2) // 2

    # Convert back to uint8
    R = R.astype(np.uint8)
    G = G.astype(np.uint8)
    B = B.astype(np.uint8)

    # Merge in BGR order (OpenCV)
    return cv2.merge([B, G, R])


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
