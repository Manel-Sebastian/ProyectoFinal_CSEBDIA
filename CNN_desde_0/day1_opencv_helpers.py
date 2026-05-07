import colorsys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans


def cargar_rgb(ruta):
    return np.array(Image.open(ruta).convert("RGB"))


def load_and_resize_rgb(ruta, scale=1.0):
    image = Image.open(ruta).convert("RGB")
    if scale != 1.0:
        new_size = (
            int(image.width * scale),
            int(image.height * scale),
        )
        image = image.resize(new_size)
    return np.array(image)


def mostrar_grid_rutas(df, n=12, seed=42, figsize=(12, 8)):
    muestra = df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)
    columnas = 4
    filas = int(np.ceil(len(muestra) / columnas))
    fig, axes = plt.subplots(filas, columnas, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, (_, fila) in zip(axes, muestra.iterrows()):
        ax.imshow(cargar_rgb(fila["ruta"]))
        ax.set_title(f'{fila["etiqueta"]} | {fila["split"]}', fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _extract_crop_for_grid(
    ruta,
    crop_kind,
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
):
    if crop_kind == "panel":
        result = debug_pipeline(
            ruta,
            panel_size=panel_size,
            target_background_gray=target_background_gray,
            target_box_gray=target_box_gray,
            dark_box_threshold=dark_box_threshold,
            hand_background_threshold=hand_background_threshold,
            hand_light_threshold=hand_light_threshold,
            panel_detection_threshold=panel_detection_threshold,
        )
        return result.get("panel"), "Sin panel"

    if crop_kind == "light_box":
        result = analyze_light_box(
            ruta,
            panel_size=panel_size,
            target_background_gray=target_background_gray,
            target_box_gray=target_box_gray,
            dark_box_threshold=dark_box_threshold,
            hand_background_threshold=hand_background_threshold,
            hand_light_threshold=hand_light_threshold,
            light_saturation_threshold=light_saturation_threshold,
            panel_detection_threshold=panel_detection_threshold,
        )
        return result.get("light_crop"), "Sin caja de luz"

    raise ValueError(f"crop_kind no soportado: {crop_kind}")


def mostrar_grid_recortes(
    df,
    crop_kind="panel",
    columns=6,
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
    title=None,
):
    muestra = df.reset_index(drop=True)
    if len(muestra) == 0:
        raise ValueError("El DataFrame no contiene imágenes para mostrar")

    columns = max(1, int(columns))
    rows = int(np.ceil(len(muestra) / columns))
    figsize = (columns * 2.6, rows * 2.6)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, (_, fila) in zip(axes, muestra.iterrows()):
        crop, missing_label = _extract_crop_for_grid(
            fila["ruta"],
            crop_kind=crop_kind,
            panel_size=panel_size,
            target_background_gray=target_background_gray,
            target_box_gray=target_box_gray,
            dark_box_threshold=dark_box_threshold,
            hand_background_threshold=hand_background_threshold,
            hand_light_threshold=hand_light_threshold,
            light_saturation_threshold=light_saturation_threshold,
            panel_detection_threshold=panel_detection_threshold,
        )

        if crop is None:
            ax.text(0.5, 0.5, missing_label, ha="center", va="center", fontsize=10)
        else:
            ax.imshow(crop)

        split = fila.get("split", "")
        split_text = f" | {split}" if split else ""
        ax.set_title(f'{fila["etiqueta"]}{split_text}', fontsize=9)
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        plt.tight_layout()
    plt.show()


def rgb_to_gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def rgb_to_hsl(rgb):
    rgb_float = rgb.astype(np.float32) / 255.0
    hls_pixels = np.array(
        [colorsys.rgb_to_hls(*pixel) for pixel in rgb_float.reshape(-1, 3)],
        dtype=np.float32,
    )
    hls = hls_pixels.reshape(*rgb.shape[:2], 3)
    return {
        "hue": hls[:, :, 0],
        "lightness": hls[:, :, 1],
        "saturation": hls[:, :, 2],
    }


def rgb_to_hsv01(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return {
        "hue": hsv[:, :, 0] / 179.0,
        "saturation": hsv[:, :, 1] / 255.0,
        "value": hsv[:, :, 2] / 255.0,
        "raw": hsv,
    }


def order_points_clockwise(points):
    points = np.asarray(points, dtype=np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    top_left = points[np.argmin(sums)]
    bottom_right = points[np.argmax(sums)]
    top_right = points[np.argmin(diffs)]
    bottom_left = points[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def draw_polygon(image, points, color=(255, 0, 0), thickness=4):
    if image is None:
        return None
    output = image.copy()
    if points is not None:
        pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)
    return output


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.35):
    if image is None or mask is None:
        return image
    output = image.copy()
    mask_bool = mask.astype(bool)
    color_array = np.array(color, dtype=np.float32)
    output[mask_bool] = (
        (1 - alpha) * output[mask_bool].astype(np.float32) + alpha * color_array
    ).astype(np.uint8)
    return output


def keep_largest_component(mask, min_pixels=0):
    labeled = ndimage.label(mask)
    if not isinstance(labeled, tuple):
        return np.zeros_like(mask, dtype=bool)

    labels, num_labels = labeled
    if num_labels == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask.astype(np.uint8), labels, index=np.arange(1, num_labels + 1))
    best_label = int(np.argmax(sizes)) + 1
    best_mask = labels == best_label
    if best_mask.sum() < min_pixels:
        return np.zeros_like(mask, dtype=bool)
    return best_mask


def extract_box_region(image, box):
    if image is None or box is None:
        return None
    x_min, y_min = box.min(axis=0)
    x_max, y_max = box.max(axis=0)
    return image[y_min:y_max, x_min:x_max]


def fit_axis_aligned_box_to_mask(mask, min_area_ratio=0.002):
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    if len(xs) < min_area_ratio * mask.size:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1
    return np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)


def normalize_with_white_background(
    rgb,
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
):
    gray = rgb_to_gray(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    white_seed = (hsv[:, :, 1] < 45) & (gray > 120)
    white_mask = ndimage.binary_opening(white_seed, structure=np.ones((7, 7), dtype=bool))
    if white_mask.mean() < 0.04:
        white_threshold = np.percentile(gray, 80)
        white_mask = gray >= white_threshold

    dark_reference_mask = gray < dark_box_threshold
    if dark_reference_mask.mean() < 0.01:
        dark_threshold = np.percentile(gray, 10)
        dark_reference_mask = gray <= dark_threshold

    background_median = float(np.median(gray[white_mask])) if np.any(white_mask) else float(np.median(gray))
    box_median = float(np.median(gray[dark_reference_mask])) if np.any(dark_reference_mask) else float(np.percentile(gray, 10))

    if background_median <= box_median + 1:
        scale = target_background_gray / max(background_median, 1.0)
        offset = 0.0
    else:
        scale = (target_background_gray - target_box_gray) / max(background_median - box_median, 1.0)
        offset = target_box_gray - scale * box_median

    normalized = np.clip(rgb.astype(np.float32) * scale + offset, 0, 255).astype(np.uint8)
    stats = {
        "background_median": background_median,
        "box_median": box_median,
        "scale": float(scale),
        "offset": float(offset),
        "target_background_gray": float(target_background_gray),
        "target_box_gray": float(target_box_gray),
        "dark_box_threshold": float(dark_box_threshold),
        "dark_reference_mask": dark_reference_mask,
    }
    return normalized, white_mask, stats


def detect_black_panel(
    rgb,
    panel_threshold=60,
    min_area_ratio=0.015,
    max_area_ratio=0.5,
):
    gray = rgb_to_gray(rgb)
    blurred = ndimage.gaussian_filter(gray, sigma=2)

    panel_seed = blurred < panel_threshold
    panel_mask = ndimage.binary_closing(panel_seed, structure=np.ones((15, 15), dtype=bool), iterations=2)
    panel_mask = ndimage.binary_opening(panel_mask, structure=np.ones((5, 5), dtype=bool))
    panel_mask = keep_largest_component(panel_mask, min_pixels=int(min_area_ratio * gray.size))

    if panel_mask.sum() == 0:
        return None, panel_mask, None
    if panel_mask.sum() > max_area_ratio * gray.size:
        return None, panel_mask, None

    contour_mask = panel_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(best_contour)
    quad = order_points_clockwise(cv2.boxPoints(rect).astype(np.float32))
    return quad, panel_mask, best_contour


def warp_panel(rgb, quad, output_size=(640, 480)):
    width, height = output_size
    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(rgb, matrix, output_size)
    return warped, matrix


def segment_hand(panel_rgb, hand_background_threshold=80, hand_light_threshold=205):
    gray = rgb_to_gray(panel_rgb)
    hsv = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2YCrCb)

    brightness_mask = gray >= hand_background_threshold
    skin_mask = (
        (ycrcb[:, :, 1] >= 133)
        & (ycrcb[:, :, 1] <= 185)
        & (ycrcb[:, :, 2] >= 77)
        & (ycrcb[:, :, 2] <= 135)
    )
    light_mask = (hsv[:, :, 1] < 80) & (gray >= hand_light_threshold)

    hand_candidate = (brightness_mask | skin_mask) & ~light_mask
    hand_candidate = ndimage.binary_closing(hand_candidate, structure=np.ones((11, 11), dtype=bool), iterations=2)
    hand_candidate = ndimage.binary_opening(hand_candidate, structure=np.ones((5, 5), dtype=bool))

    hand_mask = keep_largest_component(hand_candidate, min_pixels=1500)
    if np.any(hand_mask):
        hand_mask = ndimage.binary_closing(hand_mask, structure=np.ones((9, 9), dtype=bool))

    debug = {
        "gray": gray,
        "brightness_mask": brightness_mask,
        "skin_mask": skin_mask,
        "light_mask": light_mask,
        "hand_candidate": hand_candidate,
        "final_mask": hand_mask,
    }
    return hand_mask, debug


def orient_panel_to_left(panel, hand_mask):
    if hand_mask is None or hand_mask.sum() == 0:
        return panel, hand_mask, False
    center_x = np.where(hand_mask)[1].mean()
    if center_x <= panel.shape[1] / 2:
        return panel, hand_mask, False
    return np.ascontiguousarray(np.fliplr(panel)), np.ascontiguousarray(np.fliplr(hand_mask)), True


def debug_pipeline(
    ruta,
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    panel_detection_threshold=60,
):
    rgb = cargar_rgb(ruta)
    normalized, white_mask, normalization_stats = normalize_with_white_background(
        rgb,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
    )
    quad, panel_mask, panel_contour = detect_black_panel(normalized, panel_threshold=panel_detection_threshold)
    panel_overlay = draw_polygon(normalized, quad, color=(255, 0, 0), thickness=4)

    panel = None
    hand_mask = None
    hand_debug = {}
    flipped = False
    if quad is not None:
        panel, _ = warp_panel(normalized, quad, output_size=panel_size)
        hand_mask, hand_debug = segment_hand(
            panel,
            hand_background_threshold=hand_background_threshold,
            hand_light_threshold=hand_light_threshold,
        )
        panel, hand_mask, flipped = orient_panel_to_left(panel, hand_mask)
        hand_debug["final_mask"] = hand_mask

    return {
        "raw": rgb,
        "normalized": normalized,
        "white_mask": white_mask,
        "dark_reference_mask": normalization_stats["dark_reference_mask"],
        "panel_mask": panel_mask,
        "panel_contour": panel_contour,
        "panel_overlay": panel_overlay,
        "panel": panel,
        "hand_mask": hand_mask,
        "hand_debug": hand_debug,
        "normalization_stats": normalization_stats,
        "flipped": flipped,
    }


def detect_light_window(
    panel_rgb,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    min_area_ratio=0.002,
):
    gray = rgb_to_gray(panel_rgb)
    hsv = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2HSV)

    light_seed = (gray >= hand_light_threshold) & (hsv[:, :, 1] <= light_saturation_threshold)
    light_mask = ndimage.binary_erosion(light_seed, structure=np.ones((11, 11), dtype=bool))
    light_mask = ndimage.binary_dilation(light_mask, structure=np.ones((20, 20), dtype=bool))

    # Do not use this!!
    # light_mask = keep_largest_component(light_mask, min_pixels=int(min_area_ratio * gray.size))

    light_box = fit_axis_aligned_box_to_mask(light_mask, min_area_ratio=min_area_ratio)
    return light_box, {
        "panel_gray": gray,
        "panel_hsv": hsv,
        "light_seed": light_seed,
        "light_mask": light_mask,
        "light_box_region": extract_box_region(panel_rgb, light_box),
        "light_box_gray": extract_box_region(gray, light_box),
    }


def analyze_light_box(
    ruta,
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
):
    stage1 = debug_pipeline(
        ruta,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        panel_detection_threshold=panel_detection_threshold,
    )

    panel = stage1.get("panel")
    hand_mask = stage1.get("hand_mask")
    light_box = None
    overlay = panel
    light_crop = None
    light_crop_hand_mask = None
    light_crop_overlay = None
    light_debug = {
        "panel_gray": None,
        "panel_hsv": None,
        "light_seed": None,
        "light_mask": None,
        "light_box_region": None,
        "light_box_gray": None,
    }

    if panel is not None:
        light_box, light_debug = detect_light_window(
            panel,
            hand_light_threshold=hand_light_threshold,
            light_saturation_threshold=light_saturation_threshold,
        )
        overlay = draw_polygon(panel, light_box, color=(255, 0, 0), thickness=3)
        light_crop = extract_box_region(panel, light_box)
        light_crop_hand_mask = extract_box_region(hand_mask, light_box)
        light_crop_overlay = overlay_mask(light_crop, light_crop_hand_mask, color=(0, 180, 0), alpha=0.35)

    return {
        "stage1": stage1,
        "panel": panel,
        "hand_mask": hand_mask,
        "light_box": light_box,
        "light_debug": light_debug,
        "overlay": overlay,
        "light_crop": light_crop,
        "light_crop_hand_mask": light_crop_hand_mask,
        "light_crop_overlay": light_crop_overlay,
    }


def plot_pairwise_hsv_histograms(
    axes,
    rgb_image,
    overlay_mask=None,
    overlay_label=None,
    value_threshold=None,
    saturation_threshold=None,
    title_prefix="Panel",
):
    if rgb_image is None:
        for ax in axes:
            ax.text(0.5, 0.5, "None", ha="center", va="center", fontsize=12)
            ax.axis("off")
        return

    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hue = hsv_image[:, :, 0].ravel()
    saturation = hsv_image[:, :, 1].ravel()
    value = hsv_image[:, :, 2].ravel()

    axes[0].hist2d(hue, value, bins=[36, 32], range=[[0, 180], [0, 256]], cmap="magma")
    axes[0].set_title(f"{title_prefix} -> histograma 2D tono(H) vs luminosidad(V)")
    axes[0].set_xlabel("Tono H")
    axes[0].set_ylabel("Luminosidad V")

    axes[1].hist2d(hue, saturation, bins=[36, 32], range=[[0, 180], [0, 256]], cmap="magma")
    axes[1].set_title(f"{title_prefix} -> histograma 2D tono(H) vs saturacion(S)")
    axes[1].set_xlabel("Tono H")
    axes[1].set_ylabel("Saturacion S")

    axes[2].hist2d(saturation, value, bins=[32, 32], range=[[0, 256], [0, 256]], cmap="magma")
    axes[2].set_title(f"{title_prefix} -> histograma 2D saturacion(S) vs luminosidad(V)")
    axes[2].set_xlabel("Saturacion S")
    axes[2].set_ylabel("Luminosidad V")

    if value_threshold is not None:
        axes[0].axhline(value_threshold, color="cyan", linestyle="--", linewidth=2, label=f"V={value_threshold}")
        axes[2].axhline(value_threshold, color="cyan", linestyle="--", linewidth=2, label=f"V={value_threshold}")
    if saturation_threshold is not None:
        axes[1].axhline(saturation_threshold, color="springgreen", linestyle="--", linewidth=2, label=f"S={saturation_threshold}")
        axes[2].axvline(saturation_threshold, color="springgreen", linestyle="--", linewidth=2, label=f"S={saturation_threshold}")

    if overlay_mask is not None and np.any(overlay_mask):
        overlay_h = hsv_image[:, :, 0][overlay_mask]
        overlay_s = hsv_image[:, :, 1][overlay_mask]
        overlay_v = hsv_image[:, :, 2][overlay_mask]
        sample_step = max(len(overlay_h) // 1500, 1)
        axes[0].scatter(overlay_h[::sample_step], overlay_v[::sample_step], s=7, c="white", alpha=0.18, label=overlay_label)
        axes[1].scatter(overlay_h[::sample_step], overlay_s[::sample_step], s=7, c="white", alpha=0.18, label=overlay_label)
        axes[2].scatter(overlay_s[::sample_step], overlay_v[::sample_step], s=7, c="white", alpha=0.18, label=overlay_label)

    for ax in axes:
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best", fontsize=8)


def plot_histograms_before_after_normalization(
    ruta,
    titulo="",
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
):
    result = debug_pipeline(
        ruta,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
    )
    stats = result["normalization_stats"]
    gray_original = rgb_to_gray(result["raw"])
    gray_normalized = rgb_to_gray(result["normalized"])
    white_mask = result["white_mask"]
    dark_mask = result["dark_reference_mask"]

    background_normalized = float(np.median(gray_normalized[white_mask])) if np.any(white_mask) else float(np.median(gray_normalized))
    box_normalized = float(np.median(gray_normalized[dark_mask])) if np.any(dark_mask) else float(np.percentile(gray_normalized, 10))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes[0, 0].imshow(result["raw"])
    axes[0, 0].set_title(f"{titulo} original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(result["normalized"])
    axes[0, 1].set_title(f"{titulo} normalizada")
    axes[0, 1].axis("off")

    combined = np.zeros((*white_mask.shape, 3), dtype=np.uint8)
    combined[white_mask] = (255, 255, 255)
    combined[dark_mask] = (255, 120, 0)
    axes[0, 2].imshow(combined)
    axes[0, 2].set_title(f"Blanco y zona < {dark_box_threshold}")
    axes[0, 2].axis("off")

    axes[1, 0].hist(gray_original.ravel(), bins=64, range=(0, 255), color="steelblue", alpha=0.9)
    axes[1, 0].axvline(stats["background_median"], color="black", linestyle="--", linewidth=2, label="mediana fondo")
    axes[1, 0].axvline(stats["box_median"], color="darkorange", linestyle="--", linewidth=2, label="mediana caja")
    axes[1, 0].set_title("Histograma original")
    axes[1, 0].legend()

    axes[1, 1].hist(gray_normalized.ravel(), bins=64, range=(0, 255), color="darkorange", alpha=0.9)
    axes[1, 1].axvline(background_normalized, color="black", linestyle="--", linewidth=2, label="fondo tras normalizar")
    axes[1, 1].axvline(box_normalized, color="firebrick", linestyle="--", linewidth=2, label="caja tras normalizar")
    axes[1, 1].axvline(target_background_gray, color="tab:green", linestyle=":", linewidth=2, label="objetivo fondo")
    axes[1, 1].axvline(target_box_gray, color="tab:red", linestyle=":", linewidth=2, label="objetivo caja")
    axes[1, 1].set_title(f"Histograma normalizado | scale={stats['scale']:.3f}, offset={stats['offset']:.1f}")
    axes[1, 1].legend()

    axes[1, 2].hist(gray_original.ravel(), bins=64, range=(0, 255), color="steelblue", alpha=0.45, label="original")
    axes[1, 2].hist(gray_normalized.ravel(), bins=64, range=(0, 255), color="darkorange", alpha=0.45, label="normalizada")
    axes[1, 2].set_title("Comparacion directa")
    axes[1, 2].legend()

    for ax in axes[1]:
        ax.set_xlabel("Intensidad")
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()


def plot_box_stage(
    ruta,
    titulo="",
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
):
    result = debug_pipeline(
        ruta,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        panel_detection_threshold=panel_detection_threshold,
    )

    gray_original = rgb_to_gray(result["raw"])
    gray_normalized = rgb_to_gray(result["normalized"])
    stats = result["normalization_stats"]
    hand_debug = result.get("hand_debug") or {}
    panel_with_hand = overlay_mask(result.get("panel"), result.get("hand_mask"), color=(0, 180, 0), alpha=0.35)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.ravel()

    axes[0].imshow(result["raw"])
    axes[0].set_title(f"{titulo}\nImagen original")
    axes[0].axis("off")

    axes[1].hist(gray_original.ravel(), bins=64, range=(0, 255), color="steelblue", alpha=0.9)
    axes[1].axvline(stats["background_median"], color="black", linestyle="--", linewidth=2, label="mediana fondo")
    axes[1].axvline(stats["box_median"], color="darkorange", linestyle="--", linewidth=2, label="mediana caja")
    axes[1].set_title("Desde la original: histograma gris")
    axes[1].legend()

    axes[2].imshow(result["white_mask"], cmap="gray")
    axes[2].set_title("Original -> HSV(S<45) y gris>120\nmascara de fondo blanco")
    axes[2].axis("off")

    axes[3].imshow(result["dark_reference_mask"], cmap="gray")
    axes[3].set_title(f"Original -> gris<{dark_box_threshold}\nmascara de referencia oscura")
    axes[3].axis("off")

    axes[4].imshow(result["normalized"])
    axes[4].set_title(f"Original -> remapeo lineal\nfondo->{target_background_gray}, oscuro->{target_box_gray}")
    axes[4].axis("off")

    axes[5].hist(gray_normalized.ravel(), bins=64, range=(0, 255), color="darkorange", alpha=0.9)
    axes[5].axvline(target_background_gray, color="tab:green", linestyle=":", linewidth=2, label="objetivo fondo")
    axes[5].axvline(target_box_gray, color="tab:red", linestyle=":", linewidth=2, label="objetivo caja")
    axes[5].set_title("Tras normalizar: histograma gris")
    axes[5].legend()

    axes[6].imshow(result["panel_mask"], cmap="gray")
    axes[6].set_title(f"Normalizada -> blur gaussiano y umbral<{panel_detection_threshold}\n+ close(15), open(5) con scipy")
    axes[6].axis("off")

    axes[7].imshow(result["panel_overlay"])
    axes[7].set_title("Mascara panel -> mayor componente + minAreaRect")
    axes[7].axis("off")

    axes[8].imshow(panel_with_hand if panel_with_hand is not None else result.get("panel"))
    axes[8].set_title(f"Caja detectada -> warpPerspective{panel_size}\n+ orientar mano a la izquierda")
    axes[8].axis("off")

    axes[9].imshow(hand_debug.get("final_mask"), cmap="gray")
    axes[9].set_title(f"Panel -> (gris>={hand_background_threshold} OR piel YCrCb) AND NOT luz\n+ close(11), open(5) con scipy")
    axes[9].axis("off")

    plt.tight_layout()
    plt.show()


def plot_light_box_detection_stage(
    ruta,
    titulo="",
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
):
    result = analyze_light_box(
        ruta,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        light_saturation_threshold=light_saturation_threshold,
        panel_detection_threshold=panel_detection_threshold,
    )

    panel = result.get("panel")
    light_debug = result.get("light_debug") or {}
    light_seed = light_debug.get("light_seed")
    light_mask = light_debug.get("light_mask")
    light_overlay = result.get("overlay")
    panel_gray = light_debug.get("panel_gray")

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.ravel()

    axes[0].imshow(panel)
    axes[0].set_title(f"{titulo}\nPanel orientado")
    axes[0].axis("off")

    axes[1].imshow(light_seed, cmap="gray")
    axes[1].set_title(f"Panel -> gris>={hand_light_threshold} y S<={light_saturation_threshold}\nsemilla de luz")
    axes[1].axis("off")

    axes[2].imshow(light_mask, cmap="gray")
    axes[2].set_title("Semilla de luz -> erosion(7) + dilatacion(11)")
    axes[2].axis("off")

    axes[3].imshow(light_overlay)
    axes[3].set_title("Mascara blanca -> rectangulo axis-aligned")
    axes[3].axis("off")

    if panel_gray is None:
        axes[4].text(0.5, 0.5, "None", ha="center", va="center", fontsize=12)
        axes[4].axis("off")
    else:
        axes[4].hist(panel_gray.ravel(), bins=64, range=(0, 255), color="darkorange", alpha=0.9)
        if light_mask is not None and np.any(light_mask):
            axes[4].hist(panel_gray[light_mask].ravel(), bins=64, range=(0, 255), color="white", alpha=0.55, label="zona blanca detectada")
        axes[4].axvline(hand_light_threshold, color="cyan", linestyle="--", linewidth=2, label=f"V={hand_light_threshold}")
        axes[4].set_title("Panel -> histograma 1D de gris")
        axes[4].set_xlabel("Intensidad")
        axes[4].set_ylabel("Frecuencia")
        axes[4].legend()

    plot_pairwise_hsv_histograms(
        axes[5:8],
        panel,
        overlay_mask=light_mask,
        overlay_label="pixeles luz",
        value_threshold=hand_light_threshold,
        saturation_threshold=light_saturation_threshold,
        title_prefix="Panel",
    )

    plt.tight_layout()
    plt.show()


def analyze_hough_shadow_lines(
    light_crop,
    edge_blur_kernel=5,
    edge_canny_low=40,
    edge_canny_high=120,
    hough_threshold=35,
    hough_min_line_length=45,
    hough_max_line_gap=12,
    right_margin_fraction=0.05,
):
    if light_crop is None:
        return {
            "light_crop_gray": None,
            "light_crop_edges": None,
            "hough_lines": np.empty((0, 4), dtype=np.int32),
            "hough_overlay": None,
            "num_lines": 0,
            "right_margin_fraction": float(right_margin_fraction),
            "right_margin_x": None,
            "closest_right_gap_fraction": 1.0,
            "has_line_near_right": False,
            "pred_label": "no",
        }

    blur_kernel = max(1, int(edge_blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    light_crop_gray = cv2.cvtColor(light_crop, cv2.COLOR_RGB2GRAY)
    if blur_kernel > 1:
        light_crop_gray_blurred = cv2.GaussianBlur(light_crop_gray, (blur_kernel, blur_kernel), 0)
    else:
        light_crop_gray_blurred = light_crop_gray

    light_crop_edges = cv2.Canny(
        light_crop_gray_blurred,
        threshold1=edge_canny_low,
        threshold2=edge_canny_high,
        L2gradient=True,
    )
    raw_hough_lines = cv2.HoughLinesP(
        light_crop_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=hough_min_line_length,
        maxLineGap=hough_max_line_gap,
    )

    hough_lines = np.empty((0, 4), dtype=np.int32)
    if raw_hough_lines is not None:
        hough_lines = raw_hough_lines[:, 0].astype(np.int32)

    width = int(light_crop.shape[1])
    max_x = max(width - 1, 0)
    right_margin_x = int(np.floor((1.0 - right_margin_fraction) * max_x))
    closest_right_gap_fraction = 1.0
    has_line_near_right = False

    for x1, _y1, x2, _y2 in hough_lines:
        line_right_x = max(int(x1), int(x2))
        right_gap_fraction = (max_x - line_right_x) / max(max_x, 1)
        closest_right_gap_fraction = min(closest_right_gap_fraction, right_gap_fraction)
        if line_right_x >= right_margin_x:
            has_line_near_right = True

    hough_overlay = light_crop.copy()
    for x1, y1, x2, y2 in hough_lines:
        line_right_x = max(int(x1), int(x2))
        color = (255, 0, 0) if line_right_x >= right_margin_x else (255, 180, 0)
        cv2.line(hough_overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    cv2.line(
        hough_overlay,
        (right_margin_x, 0),
        (right_margin_x, max(light_crop.shape[0] - 1, 0)),
        (0, 255, 255),
        2,
    )

    return {
        "light_crop_gray": light_crop_gray,
        "light_crop_edges": light_crop_edges,
        "hough_lines": hough_lines,
        "hough_overlay": hough_overlay,
        "num_lines": int(len(hough_lines)),
        "right_margin_fraction": float(right_margin_fraction),
        "right_margin_x": right_margin_x,
        "closest_right_gap_fraction": float(closest_right_gap_fraction),
        "has_line_near_right": bool(has_line_near_right),
        "pred_label": "no" if has_line_near_right else "si",
    }


def analyze_light_box_hough(
    ruta,
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
    edge_blur_kernel=5,
    edge_canny_low=40,
    edge_canny_high=120,
    hough_threshold=35,
    hough_min_line_length=45,
    hough_max_line_gap=12,
    right_margin_fraction=0.05,
):
    result = analyze_light_box(
        ruta,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        light_saturation_threshold=light_saturation_threshold,
        panel_detection_threshold=panel_detection_threshold,
    )

    result["hough_debug"] = analyze_hough_shadow_lines(
        result.get("light_crop"),
        edge_blur_kernel=edge_blur_kernel,
        edge_canny_low=edge_canny_low,
        edge_canny_high=edge_canny_high,
        hough_threshold=hough_threshold,
        hough_min_line_length=hough_min_line_length,
        hough_max_line_gap=hough_max_line_gap,
        right_margin_fraction=right_margin_fraction,
    )
    return result


def plot_hand_relative_position_stage(
    ruta,
    titulo="",
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
    edge_blur_kernel=5,
    edge_canny_low=40,
    edge_canny_high=120,
    hough_threshold=35,
    hough_min_line_length=45,
    hough_max_line_gap=12,
    right_margin_fraction=0.05,
):
    result = analyze_light_box_hough(
        ruta,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        light_saturation_threshold=light_saturation_threshold,
        panel_detection_threshold=panel_detection_threshold,
        edge_blur_kernel=edge_blur_kernel,
        edge_canny_low=edge_canny_low,
        edge_canny_high=edge_canny_high,
        hough_threshold=hough_threshold,
        hough_min_line_length=hough_min_line_length,
        hough_max_line_gap=hough_max_line_gap,
        right_margin_fraction=right_margin_fraction,
    )

    light_crop = result.get("light_crop")
    light_crop_hand_mask = result.get("light_crop_hand_mask")
    light_crop_overlay = result.get("light_crop_overlay")
    hough_debug = result.get("hough_debug") or {}

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()

    if light_crop is None:
        for ax in axes[:5]:
            ax.text(0.5, 0.5, "luz no detectada", ha="center", va="center", fontsize=11, color="gray")
            ax.axis("off")
        axes[0].set_title(f"{titulo}\nCaja de luz recortada")
        for ax in axes[5:8]:
            ax.text(0.5, 0.5, "sin recorte para HSV", ha="center", va="center", fontsize=11, color="gray")
            ax.axis("off")
    else:
        light_crop_gray = hough_debug.get("light_crop_gray")
        light_crop_edges = hough_debug.get("light_crop_edges")
        light_crop_hough_overlay = hough_debug.get("hough_overlay", light_crop)
        num_lines = int(hough_debug.get("num_lines", 0))
        closest_right_gap_fraction = float(hough_debug.get("closest_right_gap_fraction", 1.0))
        pred_label = hough_debug.get("pred_label", "no")

        if light_crop_gray is None:
            light_crop_gray = cv2.cvtColor(light_crop, cv2.COLOR_RGB2GRAY)
        if light_crop_edges is None:
            light_crop_edges = np.zeros_like(light_crop_gray, dtype=np.uint8)
        if light_crop_hough_overlay is None:
            light_crop_hough_overlay = light_crop

        axes[0].imshow(light_crop)
        axes[0].set_title(f"{titulo}\nCaja de luz recortada")
        axes[0].axis("off")

        axes[1].imshow(light_crop_overlay if light_crop_overlay is not None else light_crop)
        axes[1].set_title("Caja de luz + mascara de mano\nposicion relativa dentro de la caja")
        axes[1].axis("off")

        axes[2].hist(light_crop_gray.ravel(), bins=64, range=(0, 255), color="steelblue", alpha=0.9)
        axes[2].axvline(hand_light_threshold, color="cyan", linestyle="--", linewidth=2, label=f"V={hand_light_threshold}")
        axes[2].set_title("Caja de luz -> histograma 1D\npara posicion relativa de la mano")
        axes[2].set_xlabel("Intensidad")
        axes[2].set_ylabel("Frecuencia")
        axes[2].legend()

        axes[3].imshow(light_crop_edges, cmap="gray")
        axes[3].set_title(
            "Bordes Canny\n"
            f"blur={edge_blur_kernel}, low={edge_canny_low}, high={edge_canny_high}"
        )
        axes[3].axis("off")

        axes[4].imshow(light_crop_hough_overlay)
        axes[4].set_title(
            "HoughLinesP sobre la caja\n"
            f"pred={pred_label}, n={num_lines}, gap_min={closest_right_gap_fraction:.3f}"
        )
        axes[4].axis("off")

        plot_pairwise_hsv_histograms(
            axes[5:8],
            light_crop,
            overlay_mask=light_crop_hand_mask,
            overlay_label="pixeles mano",
            value_threshold=hand_light_threshold,
            saturation_threshold=light_saturation_threshold,
            title_prefix="Caja de luz con mano",
        )

    plt.tight_layout()
    plt.show()


def plot_single_hand_summary(
    ruta,
    titulo="",
    panel_size=(640, 480),
    target_background_gray=200,
    target_box_gray=25,
    dark_box_threshold=40,
    hand_background_threshold=80,
    hand_light_threshold=205,
    light_saturation_threshold=90,
    panel_detection_threshold=60,
):
    plot_box_stage(
        ruta,
        titulo=titulo,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        light_saturation_threshold=light_saturation_threshold,
        panel_detection_threshold=panel_detection_threshold,
    )
    plot_light_box_detection_stage(
        ruta,
        titulo=titulo,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        light_saturation_threshold=light_saturation_threshold,
        panel_detection_threshold=panel_detection_threshold,
    )
    plot_hand_relative_position_stage(
        ruta,
        titulo=titulo,
        panel_size=panel_size,
        target_background_gray=target_background_gray,
        target_box_gray=target_box_gray,
        dark_box_threshold=dark_box_threshold,
        hand_background_threshold=hand_background_threshold,
        hand_light_threshold=hand_light_threshold,
        light_saturation_threshold=light_saturation_threshold,
        panel_detection_threshold=panel_detection_threshold,
    )


def analyze_banana_color(rgb):
    hsl = rgb_to_hsl(rgb)
    hsv = rgb_to_hsv01(rgb)
    gray = rgb_to_gray(rgb)

    banana_seed = (
        (hsv["hue"] > 0.10)
        & (hsv["hue"] < 0.22)
        & (hsv["saturation"] > 0.25)
        & (hsv["value"] > 0.20)
    )
    white_seed = (hsl["saturation"] < 0.12) & (hsl["lightness"] > 0.80)

    banana_mask = ndimage.binary_opening(banana_seed, structure=np.ones((5, 5), dtype=bool))
    banana_mask = ndimage.binary_closing(banana_mask, structure=np.ones((9, 9), dtype=bool))
    banana_main = keep_largest_component(banana_mask, min_pixels=50)

    white_mask = ndimage.binary_opening(white_seed, structure=np.ones((5, 5), dtype=bool))
    white_mask = ndimage.binary_closing(white_mask, structure=np.ones((9, 9), dtype=bool))

    brown_seed = banana_main & (hsv["hue"] < 0.16) & (hsv["value"] < 0.75)
    brown_mask = ndimage.binary_opening(brown_seed, structure=np.ones((3, 3), dtype=bool))
    brown_mask = ndimage.binary_closing(brown_mask, structure=np.ones((7, 7), dtype=bool))

    features_sv = np.column_stack([
        hsv["saturation"][banana_main],
        hsv["value"][banana_main],
    ])
    kmeans = KMeans(n_clusters=3, n_init="auto", random_state=0)
    labels = kmeans.fit_predict(features_sv)
    centers = kmeans.cluster_centers_
    bright_cluster = np.argmax(centers[:, 1] + 0.3 * centers[:, 0])
    bright_mask = np.zeros_like(banana_main, dtype=bool)
    bright_mask[banana_main] = labels == bright_cluster
    bright_mask = ndimage.binary_opening(bright_mask, structure=np.ones((3, 3), dtype=bool))
    bright_mask = ndimage.binary_closing(bright_mask, structure=np.ones((7, 11), dtype=bool))

    contour = None
    rotated_box = None
    if np.any(banana_main):
        contours, _ = cv2.findContours((banana_main.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        rotated_box = order_points_clockwise(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32))

    brown_box = fit_axis_aligned_box_to_mask(brown_mask)
    geometry_overlay = rgb.copy()
    if rotated_box is not None:
        cv2.drawContours(geometry_overlay, [rotated_box.astype(np.int32)], -1, (255, 0, 0), 3)
    if brown_box is not None:
        x_min, y_min = brown_box.min(axis=0)
        x_max, y_max = brown_box.max(axis=0)
        cv2.rectangle(geometry_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)

    return {
        "rgb": rgb,
        "gray": gray,
        "hsl": hsl,
        "hsv": hsv,
        "banana_seed": banana_seed,
        "white_seed": white_seed,
        "banana_mask": banana_mask,
        "banana_main": banana_main,
        "white_mask": white_mask,
        "brown_seed": brown_seed,
        "brown_mask": brown_mask,
        "features_sv": features_sv,
        "kmeans_sv": kmeans,
        "kmeans_centers": centers,
        "bright_mask": bright_mask,
        "rotated_box": rotated_box,
        "brown_box": brown_box,
        "geometry_overlay": geometry_overlay,
    }


def plot_banana_color_spaces(rgb):
    analysis = analyze_banana_color(rgb)
    hsl = analysis["hsl"]
    hsv = analysis["hsv"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(hsl["hue"], cmap="hsv")
    axes[0, 1].set_title("H de HSL")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(hsl["lightness"], cmap="gray")
    axes[0, 2].set_title("L de HSL")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(hsl["saturation"], cmap="magma")
    axes[0, 3].set_title("S de HSL")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(analysis["gray"], cmap="gray")
    axes[1, 0].set_title("Escala de grises")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hsv["hue"], cmap="hsv")
    axes[1, 1].set_title("H de HSV")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(hsv["saturation"], cmap="magma")
    axes[1, 2].set_title("S de HSV")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(hsv["value"], cmap="gray")
    axes[1, 3].set_title("V de HSV")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.show()


def plot_banana_histograms(rgb):
    analysis = analyze_banana_color(rgb)
    hsl = analysis["hsl"]
    hsv = analysis["hsv"]
    gray = analysis["gray"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].hist(gray.ravel(), bins=64, range=(0, 255), color="dimgray")
    axes[0, 0].set_title("Histograma de gris")

    axes[0, 1].hist(rgb[:, :, 0].ravel(), bins=64, range=(0, 255), color="red", alpha=0.7, label="R")
    axes[0, 1].hist(rgb[:, :, 1].ravel(), bins=64, range=(0, 255), color="green", alpha=0.5, label="G")
    axes[0, 1].hist(rgb[:, :, 2].ravel(), bins=64, range=(0, 255), color="blue", alpha=0.4, label="B")
    axes[0, 1].set_title("Histogramas RGB")
    axes[0, 1].legend()

    axes[0, 2].hist(hsl["hue"].ravel(), bins=36, range=(0, 1), color="darkorange")
    axes[0, 2].set_title("Histograma del tono H en HSL")

    axes[1, 0].hist(hsl["lightness"].ravel(), bins=50, range=(0, 1), color="slategray")
    axes[1, 0].set_title("Histograma de luminosidad L")

    axes[1, 1].hist(hsv["saturation"].ravel(), bins=50, range=(0, 1), color="purple")
    axes[1, 1].set_title("Histograma de saturación S en HSV")

    axes[1, 2].hist(hsv["value"].ravel(), bins=50, range=(0, 1), color="goldenrod")
    axes[1, 2].set_title("Histograma de valor V en HSV")

    for ax in axes.ravel():
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()


def plot_banana_2d_histograms(rgb):
    analysis = analyze_banana_color(rgb)
    hsl = analysis["hsl"]
    hsv = analysis["hsv"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist2d(hsl["hue"].ravel(), hsl["lightness"].ravel(), bins=[36, 32], range=[[0, 1], [0, 1]], cmap="magma")
    axes[0].set_title("HSL: tono vs luminosidad")
    axes[0].set_xlabel("H")
    axes[0].set_ylabel("L")

    axes[1].hist2d(hsv["hue"].ravel(), hsv["saturation"].ravel(), bins=[36, 32], range=[[0, 1], [0, 1]], cmap="magma")
    axes[1].set_title("HSV: tono vs saturación")
    axes[1].set_xlabel("H")
    axes[1].set_ylabel("S")

    axes[2].hist2d(hsv["saturation"].ravel(), hsv["value"].ravel(), bins=[32, 32], range=[[0, 1], [0, 1]], cmap="magma")
    axes[2].set_title("HSV: saturación vs valor")
    axes[2].set_xlabel("S")
    axes[2].set_ylabel("V")

    plt.tight_layout()
    plt.show()


def plot_banana_thresholds(rgb):
    analysis = analyze_banana_color(rgb)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(analysis["banana_seed"], cmap="gray")
    axes[1].set_title("Banana inicial")
    axes[1].axis("off")

    axes[2].imshow(analysis["white_seed"], cmap="gray")
    axes[2].set_title("Fondo blanco inicial")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_banana_morphology(rgb):
    analysis = analyze_banana_color(rgb)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(analysis["white_mask"], cmap="gray")
    axes[0].set_title("Fondo blanco limpio")
    axes[0].axis("off")

    axes[1].imshow(analysis["banana_mask"], cmap="gray")
    axes[1].set_title("Banana limpia")
    axes[1].axis("off")

    axes[2].imshow(analysis["banana_main"], cmap="gray")
    axes[2].set_title("Mayor componente de la banana")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_banana_brown_region(rgb):
    analysis = analyze_banana_color(rgb)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(analysis["banana_main"], cmap="gray")
    axes[0].set_title("Máscara de la banana")
    axes[0].axis("off")

    axes[1].imshow(analysis["brown_seed"], cmap="gray")
    axes[1].set_title("Semilla de zona marrón")
    axes[1].axis("off")

    axes[2].imshow(analysis["brown_mask"], cmap="gray")
    axes[2].set_title("Zona marrón tras morfología")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_banana_kmeans_sv(rgb):
    analysis = analyze_banana_color(rgb)
    centers = analysis["kmeans_centers"]
    features_sv = analysis["features_sv"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist2d(features_sv[:, 0], features_sv[:, 1], bins=[32, 32], range=[[0, 1], [0, 1]], cmap="magma")
    axes[0].scatter(centers[:, 0], centers[:, 1], c="cyan", s=80, marker="x")
    axes[0].set_title("Histograma 2D S-V dentro de la banana")
    axes[0].set_xlabel("S")
    axes[0].set_ylabel("V")

    axes[1].imshow(analysis["banana_main"], cmap="gray")
    axes[1].set_title("Banana con umbrales")
    axes[1].axis("off")

    axes[2].imshow(analysis["bright_mask"], cmap="gray")
    axes[2].set_title("Zona brillante con KMeans")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


def plot_banana_geometry(rgb):
    analysis = analyze_banana_color(rgb)
    plt.figure(figsize=(7, 8))
    plt.imshow(analysis["geometry_overlay"])
    plt.title("Banana detectada con caja rotada y zona marrón con caja axis-aligned")
    plt.axis("off")
    plt.show()


def banana_kmeans_centers_table(rgb):
    analysis = analyze_banana_color(rgb)
    return analysis["kmeans_centers"]
