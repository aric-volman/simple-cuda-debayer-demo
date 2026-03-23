import cv2


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


if __name__ == "__main__":
    main()
