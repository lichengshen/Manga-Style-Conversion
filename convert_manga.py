IMAGE_PATH = "./pictures/input_example1.jpg"
OUTPUT_PATH = "./output/output.png"
SCREENTONE_TYPES = ["dot", "grid", "sand-white", "line45", "directional-noise"]
HISTOGRAM_EQUALIZATION = False
SCREENTONE_DIR = "./output/screentones_gen"
SCREENTONE_FEATURES_DIR = "./output/screentones_features"
TARGET_LONGEST_SIDE = 2000

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform, cdist

def get_gabor_kernels(
    ksize: int = 16,
    scales: int = 4,
    orientations: int = 6,
    a: float = 1.3,
    base_sigma: float = 4.0,
    base_lambda: float = 4.0,
    gamma: float = 1.0,
):
    filters = []
    for m in range(scales):
        sigma_m = base_sigma * (a ** -m)
        lambda_m = base_lambda * (a ** -m)
        for n in range(orientations):
            theta = n * np.pi / orientations
            kernel_real = cv2.getGaborKernel(
                (ksize, ksize),
                sigma_m,
                theta,
                lambda_m,
                gamma,
                0,
                ktype=cv2.CV_32F,
            )
            kernel_imag = cv2.getGaborKernel(
                (ksize, ksize),
                sigma_m,
                theta,
                lambda_m,
                gamma,
                np.pi / 2,
                ktype=cv2.CV_32F,
            )
            filters.append((kernel_real, kernel_imag))
    return filters

def get_gabor_magnitudes(image: np.ndarray, gabor_filters):
    gabor_magnitudes = []
    for kernel_real, kernel_imag in gabor_filters:
        filtered_real = cv2.filter2D(image, cv2.CV_32F, kernel_real)
        filtered_imag = cv2.filter2D(image, cv2.CV_32F, kernel_imag)
        magnitude = np.hypot(filtered_real, filtered_imag)
        gabor_magnitudes.append(magnitude)
    return np.array(gabor_magnitudes)

def load_screentones():
    screentone_files = glob.glob(os.path.join(SCREENTONE_DIR, "*.png"))
    screentone_type_to_files = {}
    screentone_file_to_grayness = {}

    for file in screentone_files:
        name = os.path.basename(file)
        type_name = name.split("_")[0]
        if type_name not in SCREENTONE_TYPES:
            continue

        screentone_type_to_files.setdefault(type_name, []).append(file)

        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        screentone_file_to_grayness[file] = float(np.mean(img))

    return screentone_type_to_files, screentone_file_to_grayness

def get_screentone_features(
    screentone_type_to_files: dict,
    gabor_filters,
):
    os.makedirs(SCREENTONE_FEATURES_DIR, exist_ok=True)
    screentone_types = list(screentone_type_to_files.keys())
    screentone_type_features = []

    for stype in screentone_types:
        feature_path = os.path.join(SCREENTONE_FEATURES_DIR, f"{stype}.npy")

        if os.path.exists(feature_path):
            print(f"Loading features for {stype} from {SCREENTONE_FEATURES_DIR}")
            features = np.load(feature_path)
            screentone_type_features.append(features)
            continue

        features = []
        for file in screentone_type_to_files[stype]:
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            gabor_magnitudes = get_gabor_magnitudes(img, gabor_filters)

            cur_features = []
            for i in range(gabor_magnitudes.shape[0]):
                cur_features.append(np.mean(gabor_magnitudes[i]))
                cur_features.append(np.std(gabor_magnitudes[i]))
            features.append(cur_features)

        features = np.array(features, dtype=np.float32)
        features = features.mean(axis=0)
        screentone_type_features.append(features)
        np.save(feature_path, features)
        print(f"Saved features for {stype} screentones to {SCREENTONE_FEATURES_DIR}")

    screentone_type_features = np.array(screentone_type_features, dtype=np.float32)
    return screentone_types, screentone_type_features

def load_and_preprocess_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    longest_side = max(image.shape[:2])
    scale = TARGET_LONGEST_SIDE / longest_side
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if HISTOGRAM_EQUALIZATION:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        ycrcb = cv2.merge((y, cr, cb))
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    return image

def segment_image(image: np.ndarray):
    seg = cv2.ximgproc.segmentation.createGraphSegmentation(
        sigma=1, k=100, min_size=50
    )
    labels = seg.processImage(image).astype(np.int32)
    num_segments = int(labels.max()) + 1
    print(f"Segmented image into {num_segments} segments")
    return labels, num_segments

def classical_mds(pattern_feats: np.ndarray) -> np.ndarray:
    K = pattern_feats.shape[0]
    D2 = squareform(pdist(pattern_feats, metric="euclidean")) ** 2
    H = np.eye(K) - np.ones((K, K)) / K
    B = -0.5 * H.dot(D2).dot(H)
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w2 = w[idx[:2]]
    V2 = V[:, idx[:2]]
    Q = V2 * np.sqrt(w2[np.newaxis, :])
    Q -= Q.mean(axis=0)
    Q /= np.max(np.abs(Q))
    return Q

def color_to_pattern_mapping(
    image_bgr: np.ndarray,
    labels: np.ndarray,
    pattern_feats: np.ndarray,
):
    Q = classical_mds(pattern_feats)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    seg_ids = np.unique(labels)
    seg_ab = []
    for sid in seg_ids:
        mask = (labels == sid)
        a_vals = lab[:, :, 1][mask]
        b_vals = lab[:, :, 2][mask]
        seg_ab.append([a_vals.mean(), b_vals.mean()])
    seg_ab = np.array(seg_ab, dtype=np.float32)
    seg_ab -= seg_ab.mean(axis=0)
    seg_ab /= np.max(np.abs(seg_ab))
    dists = cdist(seg_ab, Q, metric="euclidean")
    best = np.argmin(dists, axis=1)
    seg_to_pattern = {sid: int(best[i]) for i, sid in enumerate(seg_ids)}

    return seg_to_pattern, Q

def get_sobel_edges(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return sobel_magnitude

def main():
    gabor_filters = get_gabor_kernels()
    screentone_type_to_files, screentone_file_to_grayness = load_screentones()    
    screentone_types, screentone_type_features = get_screentone_features(
        screentone_type_to_files,
        gabor_filters,
    )
    image = load_and_preprocess_image(IMAGE_PATH)
    labels, _ = segment_image(image)
    seg_to_pattern, _ = color_to_pattern_mapping(
        image,
        labels,
        screentone_type_features,
    )

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32) / 255.0
    h, w = labels.shape
    output_image = np.full((h, w), 255, dtype=np.uint8)

    for seg_id, pattern_idx in tqdm(seg_to_pattern.items(), desc="Assigning screentones"):
        mask = (labels == seg_id)
        L_vals = L[mask]
        L_mean = float(np.mean(L_vals))

        best_dist = np.inf
        best_file = None
        pattern_name = screentone_types[pattern_idx]

        for file in screentone_type_to_files[pattern_name]:
            grayness = screentone_file_to_grayness[file]
            dist = abs(grayness - L_mean)
            if dist < best_dist:
                best_dist = dist
                best_file = file

        screentone_image = cv2.imread(best_file, cv2.IMREAD_GRAYSCALE)
        screentone_image = screentone_image[:h, :w]

        output_image[mask] = screentone_image[mask]

    sobel_edges = get_sobel_edges(image)
    sobel_edges = (sobel_edges - sobel_edges.min()) / (
        sobel_edges.max() - sobel_edges.min()
    )
    edge_mask = (sobel_edges < 0.2).astype(np.uint8) * 255
    final = cv2.bitwise_and(output_image, edge_mask)

    cv2.imwrite(OUTPUT_PATH, final)
    print(f"Saved final image to {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert image to manga style using screentones.")
    parser.add_argument("--input_image", type=str, default=IMAGE_PATH, help="Path to input image.")
    parser.add_argument("--output_image", type=str, default=OUTPUT_PATH, help="Path to output image.")
    parser.add_argument("--screentone_dir", type=str, default=SCREENTONE_DIR, help="Directory containing screentone images.")
    parser.add_argument("--screentone_features_dir", type=str, default=SCREENTONE_FEATURES_DIR, help="Directory to save/load screentone features.")
    args = parser.parse_args()
    IMAGE_PATH = args.input_image
    OUTPUT_PATH = args.output_image
    SCREENTONE_DIR = args.screentone_dir
    SCREENTONE_FEATURES_DIR = args.screentone_features_dir
    main()
