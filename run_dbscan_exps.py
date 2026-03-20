from tqdm.auto import tqdm

from src.biohack.utils import load_grayscale_image
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from src.biohack.utils import load_image_paths
@dataclass
class ImageSample:
    path: Path
    original_image: np.ndarray
    blacked_image: np.ndarray


paths = load_image_paths(r"data/separated_frames/")
filtered_imgs = []

import time
from src.biohack.utils import set_lowest_percent_to_black

from IPython.display import clear_output

for i in range(len(paths)):
# for i in range(len(paths))[:1]:
    img_test = load_grayscale_image(paths[i])
    print(f"Image {i}, shape: {paths[i].name}")
    # matrix_to_image(img_test, show=True)

    img_blackened = set_lowest_percent_to_black(img_test, black_percent=99.5)
    # matrix_to_image(img_blackened, show=False)
    filtered_imgs.append(ImageSample(paths[i], img_test, img_blackened))
    # time.sleep(.2)
    # clear_output(wait=True)



eps_search_space = [0.5,1,2,5]
min_samples_search_space = [5,10,20,30]

for eps in tqdm(eps_search_space):
    for min_samples in tqdm(min_samples_search_space):
        for i in tqdm(range(len(filtered_imgs)), leave=False):
            n = i
            # _=matrix_to_image(filtered_imgs[n].original_image)
            # _=matrix_to_image(filtered_imgs[n].blacked_image)

            import numpy as np
            from sklearn.cluster import DBSCAN

            # cleaned: your cleaned 2D numpy image (uint8/float), e.g. after blacking low pixels
            # Example: cleaned = img_clean
            cleaned = filtered_imgs[n].blacked_image

            # 1) Keep only non-black pixels
            mask = cleaned > 0
            rows, cols = np.where(mask)

            # 2) Build DBSCAN features
            # spatial coords
            x = cols.astype(np.float32)
            y = rows.astype(np.float32)

            # optional: include intensity as 3rd feature
            intensity = cleaned[rows, cols].astype(np.float32)
            intensity = intensity / 255.0   # keep scale comparable
            intensity_weight = 0.3          # tune this
            intensity = intensity * intensity_weight

            X = np.column_stack([x, y, intensity])   # or np.column_stack([x, y]) for spatial only

            # 3) Run DBSCAN
            db = DBSCAN(eps=2.0, min_samples=5)      # tune eps/min_samples
            labels = db.fit_predict(X)

            # 4) Put labels back into image shape for plotting
            cluster_map = np.full(cleaned.shape, -1, dtype=np.int32)
            cluster_map[rows, cols] = labels

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = np.sum(labels == -1)
            # print(f"clusters={n_clusters}, noise={n_noise}")


            import matplotlib.pyplot as plt

            # Build an RGB image for cluster visualization
            # noise (-1) stays black; each cluster id gets a random color
            cluster_rgb = np.zeros((*cluster_map.shape, 3), dtype=np.uint8)

            valid = labels >= 0
            unique_labels = np.unique(labels[valid])
            rng = np.random.default_rng(42)
            colors = rng.integers(40, 255, size=(len(unique_labels), 3), dtype=np.uint8)

            for idx, label in enumerate(unique_labels):
                pix = labels == label
                cluster_rgb[rows[pix], cols[pix]] = colors[idx]

            # plt.figure(figsize=(5, 5))
            # plt.imshow(cluster_rgb)
            # plt.title(f"DBSCAN clusters: {n_clusters} (noise: {n_noise})")
            # plt.axis("off")
            # plt.show()

            # Plot original, blackened, and clustered image side-by-side and save
            output_dir = Path(rf"outputs/dbscan_eps{eps}_min_samples{min_samples}")  # change to your specific folder
            output_dir.mkdir(parents=True, exist_ok=True)

            sample = filtered_imgs[n]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(sample.original_image, cmap="gray")
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(sample.blacked_image, cmap="gray")
            axes[1].set_title("Blackened")
            axes[1].axis("off")

            axes[2].imshow(cluster_rgb)
            axes[2].set_title(f"Clustered ({n_clusters} clusters)")
            axes[2].axis("off")

            fig.suptitle(sample.path.name)
            fig.tight_layout()

            save_path = output_dir / f"{sample.path.stem}_overview.png"
            _ = fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            # plt.show()

            # print(f"Saved: {save_path}")


