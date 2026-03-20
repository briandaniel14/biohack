import numpy as np
from PIL import Image


def load_tiffs(tiff_paths: list[str]) -> list[list[np.array]]:
    movies = []

    for path in tiff_paths:
        img = Image.open(path)
        frames = []

        try:
            while True:
                frames.append(np.array(img))
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # No more frames

        movies.append(frames)

    return movies
