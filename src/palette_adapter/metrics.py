from typing import Any, List, Tuple, cast, Callable

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.metrics import ndcg_score  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore


from ip_adapter_palette.types import Color, Palette, BatchOutput, ImageAndPalette

from refiners.fluxion.utils import tensor_to_images



def image_palette_metrics(
    image: Image.Image, palette: list[Color], img_size: Tuple[int, int] = (256, 256), sampling_size: int = 1000
):
    resized_img = image.resize(img_size)
    Point = npt.NDArray[np.float64]
    all_points: List[Point] = np.array(resized_img.getdata(), dtype=np.float64)  # type: ignore
    choices = np.random.choice(len(all_points), sampling_size)
    points = all_points[choices]

    num = len(palette)

    centroids = np.stack(palette)

    nn = NearestNeighbors(n_neighbors=num)
    nn.fit(centroids)  # type: ignore

    indices: npt.NDArray[np.int8] = nn.kneighbors(points, return_distance=False)  # type: ignore
    indices = indices[:, 0]

    counts = np.bincount(indices)  # type: ignore
    counts = np.pad(counts, (0, num - len(counts)), "constant")  # type: ignore
    y_true_ranking = list(range(num, 0, -1))

    distances_list: List[float] = []

    def distance(a: Point, b: Point) -> float:
        return np.linalg.norm(a - b).item()

    for i in range(len(centroids)):
        condition = np.where(indices == i)

        cluster_points = points[condition]
        distances = [distance(p, centroids[i]) for p in cluster_points]
        distances_list.extend(distances)

    return ([y_true_ranking], [counts], distances_list)


def batch_image_palette_metrics(log: Callable[[Any], None], images_and_palettes: list[ImageAndPalette], prefix: str = "palette-img"):
    per_num: dict[int, Any] = {}
    for image_and_palette in images_and_palettes:
        palette = image_and_palette["palette"]
        image = image_and_palette["image"]
        num = len(palette)

        (y_true_ranking, counts, distances_list) = image_palette_metrics(image, palette)
        if not num in per_num:
            per_num[num] = {
                "y_true_ranking": y_true_ranking,
                "counts": counts,
                "distances": distances_list,
            }
        else:
            per_num[num]["y_true_ranking"] += y_true_ranking
            per_num[num]["counts"] += counts
            per_num[num]["distances"] += distances_list

    for num in per_num:
        if num > 1:
            score: float = ndcg_score(per_num[num]["y_true_ranking"], per_num[num]["counts"]).item()
            log({f"{prefix}/ndcg_{num}": score, f"{prefix}/std_dev_{num}": np.std(per_num[num]["distances"]).item()})
        else:
            log({f"{prefix}/std_dev_{num}": np.std(per_num[num]["distances"]).item()})



def batch_palette_metrics(log: Callable[[Any], None], images_and_palettes: BatchOutput, prefix: str = "palette-img"):
    
    source_palettes = cast(list[Palette], images_and_palettes.source_palettes) # type: ignore
    palettes: list[list[Color]] = []
    for p in source_palettes: # type: ignore
        colors : list[Color] = []
        sorted_clusters = sorted(p, key=lambda x: x[1], reverse=True)
        for sorted_clusters in p:
            colors.append(sorted_clusters[0])
        palettes.append(colors)
    
    images = tensor_to_images(images_and_palettes.result_images) # type: ignore
    
    if len(images) != len(palettes):
        raise ValueError("Images and palettes must have the same length")
        
    return batch_image_palette_metrics(
        log,
        [
            ImageAndPalette({"image": image, "palette": palette})
            for image, palette in zip(images, palettes)
        ],
        prefix
    )