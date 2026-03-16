#!/usr/bin/env python3

# ----- Import libraries, global settings -----

from math import floor, sqrt
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple
from PIL import Image

DEBUG_LOGGING = False

if DEBUG_LOGGING:
    import binascii
    import struct
try:
    import numpy as np
    USE_NUMPY = True
except ImportError:
    USE_NUMPY = False

try:
    import faiss  # type: ignore
    USE_FAISS = True
except ImportError:
    faiss = None
    USE_FAISS = False

# ----- Helper -----


def clamp(val, min_, max_):
    return max(min_, min(max_, val))


# ----- Extracted constants -----

# These constants are used as weights for each differently-sized
# rectangle during the feature extraction phase.
# This is used in Equation 11 in the paper.
WEIGHT_R1 = float.fromhex('0x1.936398bf0aae3p-3')
WEIGHT_R2 = float.fromhex('0x1.caddcd96f4881p-2')
WEIGHT_R3 = float.fromhex('0x1.1cb5cf620ef1dp-2')

# This is used for initial hash scaling.
# This is described in section 3.4 of the paper.
HASH_SCALE_CONST = float.fromhex('0x1.07b3705abb25cp0')

# This parameter is used to switch between "robust" and "short"
# hashes. It is not clear how exactly this is intended to be used
# (e.g. "short" hashes have a totally different postprocessing step).
# The only value used in practice is 6. Changing it may or may not work.
GRID_SIZE_HYPERPARAMETER = 6

# ----- (3.1) Preprocessing -----

# Compute the summed pixel data. The summed data has the same dimensions
# as the input image. For each pixel position, the output at that point
# is the sum of all pixels in the rectangle from the origin over to
# the given point. The RGB channels are summed together.
def preprocess_pixel_sum(im):
    sum_out = []

    # The first row does not have a row above it, so we treat it specially
    accum = 0
    for x in range(im.width):
        px = im.getpixel((x, 0))
        # Sum RGB channels
        pxsum = px[0] + px[1] + px[2]
        # As the x coordinate moves right, we sum up everything
        # starting from the beginning of the row.
        accum += pxsum
        sum_out.append(accum)

    # For all subsequent rows, there is a row above.
    # We can save a lot of processing time by reusing that information.
    # (This is a straightforward example of "dynamic programming".)
    for y in range(1, im.height):
        accum = 0
        for x in range(im.width):
            px = im.getpixel((x, y))
            # Sum RGB channels
            pxsum = px[0] + px[1] + px[2]
            # `accum` is the sum of just this row
            accum += pxsum
            # Re-use already-computed data from previous row
            last_row_sum = sum_out[(y - 1) * im.width + x]
            sum_out.append(accum + last_row_sum)

    return sum_out


# Optimized implementation using NumPy
def preprocess_pixel_sum_np(im):
    # Convert to NumPy
    im = np.array(im, dtype=np.uint64)
    # Sum RGB components
    im = im.sum(axis=2)
    # Sum along each row ("over" columns)
    im = im.cumsum(axis=1)
    # Sum down the image ("over" rows)
    im = im.cumsum(axis=0)
    return im.flatten()


# ----- (3.2) Feature extraction -----

# This is equal to 26. This means that the `u` and `v` coordinates
# mentioned in the paper both range from [0, 25].
FEATURE_GRID_DIM = GRID_SIZE_HYPERPARAMETER * 4 + 2

# This is used to compute the step size which maps
# from grid points to image points. (It is not the step size itself.)
# This is slightly bigger than the feature grid dimensions in order to
# make each region overlap slightly.
FEATURE_STEP_DIVISOR = GRID_SIZE_HYPERPARAMETER * 4 + 4


# This is Equation 9 in the paper. It performs bilinear interpolation.
# The purpose of this is to better approximate the pixel information
# at a coordinate which is not an integer (and thus lies *between* pixels).
def interpolate_px_quad(summed_im, im_w, x, y, x_residue, y_residue, debug_str=''):
    px_1 = summed_im[y * im_w + x]
    px_2 = summed_im[(y + 1) * im_w + x]
    px_3 = summed_im[y * im_w + x + 1]
    px_4 = summed_im[(y + 1) * im_w + x + 1]
    # NOTE: Must multiply the interpolation factors first *and then* the pixel
    # (due to rounding behavior)
    px_avg = \
        ((1 - x_residue) * (1 - y_residue) * px_1) + \
        ((1 - x_residue) * y_residue * px_2) + \
        (x_residue * (1 - y_residue) * px_3) + \
        (x_residue * y_residue * px_4)
    if DEBUG_LOGGING:
        print(f"px {debug_str} {px_1} {px_2} {px_3} {px_4} | {px_avg}")
    return px_avg


# This eventually computes Equation 10 in the paper.
# This "box sum" is a blurred average over regions of the image.
def box_sum_for_radius(
        summed_im, im_w, im_h,
        grid_step_h, grid_step_v,
        grid_point_x, grid_point_y,
        radius, weight):

    # Compute where the corners are. This is Equation 6.
    # NOTE: Parens required for rounding.
    corner_a_x = grid_point_x + (- radius * grid_step_h - 1)
    corner_a_y = grid_point_y + (- radius * grid_step_v - 1)
    corner_d_x = grid_point_x + radius * grid_step_h
    corner_d_y = grid_point_y + radius * grid_step_v
    # Make sure the corners are within the image bounds
    corner_a_x = clamp(corner_a_x, 0, im_w - 2)
    corner_a_y = clamp(corner_a_y, 0, im_h - 2)
    corner_d_x = clamp(corner_d_x, 0, im_w - 2)
    corner_d_y = clamp(corner_d_y, 0, im_h - 2)
    if DEBUG_LOGGING:
        print(f"corner r{radius} {corner_a_x} {corner_a_y} | {corner_d_x} {corner_d_y}")

    # Get an integer pixel coordinate for the corners.
    # This is Equation 7.
    corner_a_x_int = int(corner_a_x)
    corner_a_y_int = int(corner_a_y)
    corner_d_x_int = int(corner_d_x)
    corner_d_y_int = int(corner_d_y)
    # Compute the fractional part, since we need it for interpolation.
    # This is Equation 8.
    corner_a_x_residue = corner_a_x - corner_a_x_int
    corner_a_y_residue = corner_a_y - corner_a_y_int
    corner_d_x_residue = corner_d_x - corner_d_x_int
    corner_d_y_residue = corner_d_y - corner_d_y_int
    if DEBUG_LOGGING:
        print(f"corner int r{radius} {corner_a_x_int} {corner_a_y_int} | {corner_d_x_int} {corner_d_y_int}")

    # Fetch the pixels in each corner
    px_A = interpolate_px_quad(
        summed_im,
        im_w,
        corner_a_x_int,
        corner_a_y_int,
        corner_a_x_residue,
        corner_a_y_residue,
        f"r{radius} A")
    px_B = interpolate_px_quad(
        summed_im,
        im_w,
        corner_d_x_int,
        corner_a_y_int,
        corner_d_x_residue,
        corner_a_y_residue,
        f"r{radius} B")
    px_C = interpolate_px_quad(
        summed_im,
        im_w,
        corner_a_x_int,
        corner_d_y_int,
        corner_a_x_residue,
        corner_d_y_residue,
        f"r{radius} C")
    px_D = interpolate_px_quad(
        summed_im,
        im_w,
        corner_d_x_int,
        corner_d_y_int,
        corner_d_x_residue,
        corner_d_y_residue,
        f"r{radius} D")

    # Compute the final sum. This is Equation 10 and 11, rearranged.
    # NOTE: The computation needs to be performed like this for rounding to match.
    R_box = px_A * weight - px_B * weight - px_C * weight + px_D * weight
    if DEBUG_LOGGING:
        print(f"box sum r{radius} {R_box}")
    return R_box


def compute_feature_grid(summed_im, im_w, im_h):
    # Compute the grid step size, which is Delta_l and Delta_w in the paper.
    # The paper does not explain how to do this.
    grid_step_h = im_w / FEATURE_STEP_DIVISOR
    grid_step_v = im_h / FEATURE_STEP_DIVISOR
    if DEBUG_LOGGING:
        print(f"step {grid_step_h} {grid_step_v}")

    feature_grid = [0.0] * (FEATURE_GRID_DIM * FEATURE_GRID_DIM)
    for feat_y in range(FEATURE_GRID_DIM):
        for feat_x in range(FEATURE_GRID_DIM):
            if DEBUG_LOGGING:
                print(f"-- grid {feat_x} {feat_y} --")

            # Compute what pixel the feature grid point maps to in the source image.
            # This is Equation 5 in the paper. The value of zeta is 1.5.
            grid_point_x = (feat_x + 1.5) * grid_step_h
            grid_point_y = (feat_y + 1.5) * grid_step_v
            if DEBUG_LOGGING:
                print(f"grid point {grid_point_x} {grid_point_y}")

            # Compute the box sum for each radius.
            # The radii scaling factors are 0.2, 0.4, and 0.8.
            radius_box_0p2 = box_sum_for_radius(
                summed_im,
                im_w, im_h,
                grid_step_h, grid_step_v,
                grid_point_x, grid_point_y,
                0.2, WEIGHT_R1)
            radius_box_0p4 = box_sum_for_radius(
                summed_im,
                im_w, im_h,
                grid_step_h, grid_step_v,
                grid_point_x, grid_point_y,
                0.4, WEIGHT_R2)
            radius_box_0p8 = box_sum_for_radius(
                summed_im,
                im_w, im_h,
                grid_step_h, grid_step_v,
                grid_point_x, grid_point_y,
                0.8, WEIGHT_R3)

            # Compute the final feature value. This is Equation 11.
            feat_val = radius_box_0p2 + radius_box_0p4 + radius_box_0p8
            if DEBUG_LOGGING:
                print(f"--> {feat_val}")
            feature_grid[feat_y * FEATURE_GRID_DIM + feat_x] = feat_val

    return (feature_grid, grid_step_h, grid_step_v)


# ----- (3.3) Gradient processing -----

def compute_gradient_grid(feature_grid):
    grad_out = [0.0] * (GRID_SIZE_HYPERPARAMETER * GRID_SIZE_HYPERPARAMETER * 4)
    for feat_y_chunk in range(GRID_SIZE_HYPERPARAMETER):
        for feat_x_chunk in range(GRID_SIZE_HYPERPARAMETER):
            for feat_chunk_sub_y in range(4):
                for feat_chunk_sub_x in range(4):
                    feat_x = 1 + feat_x_chunk * 4 + feat_chunk_sub_x
                    feat_y = 1 + feat_y_chunk * 4 + feat_chunk_sub_y
                    if DEBUG_LOGGING:
                        print(f"feat {feat_x} {feat_y}")

                    feat_L = feature_grid[feat_y * FEATURE_GRID_DIM + feat_x - 1]
                    feat_R = feature_grid[feat_y * FEATURE_GRID_DIM + feat_x + 1]
                    feat_U = feature_grid[(feat_y - 1) * FEATURE_GRID_DIM + feat_x]
                    feat_D = feature_grid[(feat_y + 1) * FEATURE_GRID_DIM + feat_x]
                    if DEBUG_LOGGING:
                        print(f"vals {feat_L} {feat_R} {feat_U} {feat_D}")

                    grad_d_horiz = feat_L - feat_R
                    grad_d_vert = feat_U - feat_D

                    if grad_d_horiz <= 0:
                        grad_d_h_pos = 0
                        grad_d_h_neg = -grad_d_horiz
                    else:
                        grad_d_h_pos = grad_d_horiz
                        grad_d_h_neg = 0
                    if grad_d_vert <= 0:
                        grad_d_v_pos = 0
                        grad_d_v_neg = -grad_d_vert
                    else:
                        grad_d_v_pos = grad_d_vert
                        grad_d_v_neg = 0

                    if DEBUG_LOGGING:
                        print(
                            f"grad values {binascii.hexlify(struct.pack('>d', grad_d_horiz))} "
                            f"{binascii.hexlify(struct.pack('>d', grad_d_vert))}"
                        )

                    grad_y_f = (feat_y - 2.5) * 0.25
                    grad_x_f = (feat_x - 2.5) * 0.25
                    grad_y = floor(grad_y_f)
                    grad_x = floor(grad_x_f)
                    grad_y_residue = grad_y_f - grad_y
                    grad_x_residue = grad_x_f - grad_x
                    if DEBUG_LOGGING:
                        print(f"grad pos {grad_x} {grad_y} | {grad_x_residue} {grad_y_residue}")

                    if grad_y >= 0:
                        if grad_x >= 0:
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 0] += \
                                (1 - grad_x_residue) * (1 - grad_y_residue) * grad_d_h_pos
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 1] += \
                                (1 - grad_x_residue) * (1 - grad_y_residue) * grad_d_h_neg
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 2] += \
                                (1 - grad_x_residue) * (1 - grad_y_residue) * grad_d_v_pos
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 3] += \
                                (1 - grad_x_residue) * (1 - grad_y_residue) * grad_d_v_neg
                        if grad_x < 5:
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 0] += \
                                grad_x_residue * (1 - grad_y_residue) * grad_d_h_pos
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 1] += \
                                grad_x_residue * (1 - grad_y_residue) * grad_d_h_neg
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 2] += \
                                grad_x_residue * (1 - grad_y_residue) * grad_d_v_pos
                            grad_out[(grad_y * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 3] += \
                                grad_x_residue * (1 - grad_y_residue) * grad_d_v_neg
                    if grad_y < 5:
                        if grad_x >= 0:
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 0] += \
                                (1 - grad_x_residue) * grad_y_residue * grad_d_h_pos
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 1] += \
                                (1 - grad_x_residue) * grad_y_residue * grad_d_h_neg
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 2] += \
                                (1 - grad_x_residue) * grad_y_residue * grad_d_v_pos
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x) * 4 + 3] += \
                                (1 - grad_x_residue) * grad_y_residue * grad_d_v_neg
                        if grad_x < 5:
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 0] += \
                                grad_x_residue * grad_y_residue * grad_d_h_pos
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 1] += \
                                grad_x_residue * grad_y_residue * grad_d_h_neg
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 2] += \
                                grad_x_residue * grad_y_residue * grad_d_v_pos
                            grad_out[((grad_y + 1) * GRID_SIZE_HYPERPARAMETER + grad_x + 1) * 4 + 3] += \
                                grad_x_residue * grad_y_residue * grad_d_v_neg

    return grad_out


# ----- (3.4) Hash normalization -----

HASH_ITER_LIMIT = 10
HASH_CLIP_CONST = 0.25


def process_hash(gradient_grid, grid_step_h, grid_step_v):
    scale_factor = grid_step_h * HASH_SCALE_CONST * grid_step_v * 3
    for i in range(len(gradient_grid)):
        gradient_grid[i] /= scale_factor

    iter_count = 0
    while iter_count < HASH_ITER_LIMIT:
        did_clip = False
        iter_count += 1

        l2_norm = 1e-8
        for i in range(len(gradient_grid)):
            l2_norm += gradient_grid[i] * gradient_grid[i]
        l2_norm = sqrt(l2_norm)

        if DEBUG_LOGGING:
            print(f"iter {iter_count}, norm {l2_norm}")

        for i in range(len(gradient_grid)):
            val_i = gradient_grid[i] / l2_norm
            gradient_grid[i] = val_i

            if val_i >= HASH_CLIP_CONST and iter_count < HASH_ITER_LIMIT:
                if DEBUG_LOGGING:
                    print(f"idx {i} clipped")
                gradient_grid[i] = HASH_CLIP_CONST
                did_clip = True

        if not did_clip:
            break
    if DEBUG_LOGGING:
        print("iter done!")

    return gradient_grid


# This is Equation 17 in the paper
def hash_to_bytes(hash_in):
    hash_out = []
    for i in range(len(hash_in)):
        b = hash_in[i] * 256 / HASH_CLIP_CONST
        b = clamp(b, 0, 255)
        b = int(b)
        hash_out.append(b)
    return hash_out


# ----- Hash comparison helpers -----

def compute_hash(filename):
    im = Image.open(filename)
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    if not USE_NUMPY:
        summed_pixels = preprocess_pixel_sum(im)
    else:
        summed_pixels = preprocess_pixel_sum_np(im)
    (feature_grid, grid_step_h, grid_step_v) = compute_feature_grid(summed_pixels, im.width, im.height)
    gradient_grid = compute_gradient_grid(feature_grid)
    hash_as_floats = process_hash(gradient_grid, grid_step_h, grid_step_v)
    hash_as_bytes = hash_to_bytes(hash_as_floats)
    return hash_as_bytes


def hash_dimension() -> int:
    return GRID_SIZE_HYPERPARAMETER * GRID_SIZE_HYPERPARAMETER * 4


def max_euclidean_distance(dim: int) -> float:
    return sqrt(dim * (255 ** 2))


def squared_l2_to_euclidean(squared_l2: float) -> float:
    return sqrt(max(0.0, squared_l2))


def euclidean_to_similarity(distance: float, dim: int) -> float:
    similarity = 1.0 - (distance / max_euclidean_distance(dim))
    return clamp(similarity, 0.0, 1.0)


def squared_l2_to_similarity(squared_l2: float, dim: int) -> float:
    return euclidean_to_similarity(squared_l2_to_euclidean(squared_l2), dim)


def similarity_to_max_squared_l2(similarity: float, dim: int) -> float:
    similarity = clamp(similarity, 0.0, 1.0)
    max_dist = max_euclidean_distance(dim)
    max_allowed_dist = (1.0 - similarity) * max_dist
    return max_allowed_dist * max_allowed_dist


def hash_to_vector(hash_values):
    if USE_NUMPY:
        return np.asarray(hash_values, dtype=np.float32)
    return [float(x) for x in hash_values]


def compare_hashes(hash1, hash2, metric='euclidean'):
    if len(hash1) != len(hash2):
        raise ValueError('Hashes must have the same length')

    if metric == 'euclidean':
        return sqrt(sum((a - b) ** 2 for a, b in zip(hash1, hash2)))
    if metric == 'manhattan':
        return sum(abs(a - b) for a, b in zip(hash1, hash2))

    raise ValueError(f'Unsupported metric: {metric}')


def similarity_score(hash1, hash2):
    distance = compare_hashes(hash1, hash2, metric='euclidean')
    return euclidean_to_similarity(distance, len(hash1))


def compare_images(file1, file2, metric='euclidean'):
    hash1 = compute_hash(file1)
    hash2 = compute_hash(file2)
    distance = compare_hashes(hash1, hash2, metric=metric)
    return {
        'file1': file1,
        'file2': file2,
        'metric': metric,
        'distance': distance,
        'similarity': similarity_score(hash1, hash2),
        'hash1': hash1,
        'hash2': hash2,
    }


# ----- FAISS local DB helpers -----

def require_faiss():
    if not USE_FAISS:
        raise RuntimeError('FAISS is not installed. Install it with: pip install faiss-cpu')


def default_meta(dim: int) -> Dict:
    return {
        'dimension': dim,
        'metric': 'squared_l2',
        'similarity_metric': 'normalized_euclidean',
        'next_id': 1,
        'items': []
    }


def load_meta(meta_path: str, dim: int) -> Dict:
    if not os.path.exists(meta_path):
        return default_meta(dim)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    if meta.get('dimension') != dim:
        raise ValueError(f"Metadata dimension mismatch: expected {dim}, got {meta.get('dimension')}")
    return meta


def save_meta(meta_path: str, meta: Dict):
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def create_faiss_index(dim: int):
    require_faiss()
    return faiss.IndexIDMap2(faiss.IndexFlatL2(dim))


def load_faiss_index(index_path: str, dim: int):
    require_faiss()
    if not os.path.exists(index_path):
        return create_faiss_index(dim)
    index = faiss.read_index(index_path)
    if index.d != dim:
        raise ValueError(f"FAISS index dimension mismatch: expected {dim}, got {index.d}")
    return index


def save_faiss_index(index_path: str, index):
    require_faiss()
    faiss.write_index(index, index_path)


def build_records_for_files(files: List[str], meta: Dict) -> Tuple[List[int], List[List[int]], List[Dict]]:
    ids = []
    hashes = []
    items = []
    next_id = meta['next_id']

    existing_paths = {item['path'] for item in meta['items']}
    for path in files:
        if path in existing_paths:
            continue
        h = compute_hash(path)
        item_id = next_id
        next_id += 1
        ids.append(item_id)
        hashes.append(h)
        items.append({
            'id': item_id,
            'path': path,
            'hash': h,
            'extra': {}
        })

    meta['next_id'] = next_id
    return ids, hashes, items


def add_files_to_faiss(index_path: str, meta_path: str, files: List[str]):
    dim = hash_dimension()
    meta = load_meta(meta_path, dim)
    index = load_faiss_index(index_path, dim)
    ids, hashes, items = build_records_for_files(files, meta)

    if not ids:
        save_faiss_index(index_path, index)
        save_meta(meta_path, meta)
        return 0

    if USE_NUMPY:
        xb = np.asarray(hashes, dtype=np.float32)
        xids = np.asarray(ids, dtype=np.int64)
    else:
        raise RuntimeError('NumPy is required for FAISS operations')

    index.add_with_ids(xb, xids)
    meta['items'].extend(items)
    save_faiss_index(index_path, index)
    save_meta(meta_path, meta)
    return len(ids)


def build_faiss_index(index_path: str, meta_path: str, files: List[str]):
    dim = hash_dimension()
    meta = default_meta(dim)
    index = create_faiss_index(dim)
    ids, hashes, items = build_records_for_files(files, meta)

    if ids:
        if not USE_NUMPY:
            raise RuntimeError('NumPy is required for FAISS operations')
        xb = np.asarray(hashes, dtype=np.float32)
        xids = np.asarray(ids, dtype=np.int64)
        index.add_with_ids(xb, xids)
        meta['items'].extend(items)

    save_faiss_index(index_path, index)
    save_meta(meta_path, meta)
    return len(ids)


def query_faiss_index(index_path: str,
                      meta_path: str,
                      query_file: str,
                      top_k: int = 10,
                      min_similarity: Optional[float] = None,
                      max_distance: Optional[float] = None):
    dim = hash_dimension()
    meta = load_meta(meta_path, dim)
    index = load_faiss_index(index_path, dim)
    if index.ntotal == 0:
        return {
            'query_file': query_file,
            'query_hash': compute_hash(query_file),
            'results': []
        }

    query_hash = compute_hash(query_file)
    if not USE_NUMPY:
        raise RuntimeError('NumPy is required for FAISS operations')

    xq = np.asarray([query_hash], dtype=np.float32)
    search_k = min(max(top_k, 1), int(index.ntotal))

    # If filtering is requested, oversample so the filter still has a good chance
    # of returning enough close matches.
    if min_similarity is not None or max_distance is not None:
        search_k = int(index.ntotal)

    distances_sq, ids = index.search(xq, search_k)
    item_by_id = {item['id']: item for item in meta['items']}

    if min_similarity is not None:
        min_similarity = clamp(float(min_similarity), 0.0, 1.0)
    if max_distance is not None:
        max_distance = max(0.0, float(max_distance))

    results = []
    for squared_l2, item_id in zip(distances_sq[0], ids[0]):
        if item_id == -1:
            continue
        distance = squared_l2_to_euclidean(float(squared_l2))
        similarity = squared_l2_to_similarity(float(squared_l2), dim)

        if min_similarity is not None and similarity < min_similarity:
            continue
        if max_distance is not None and distance > max_distance:
            continue

        item = item_by_id.get(int(item_id))
        if item is None:
            continue
        results.append({
            'id': int(item_id),
            'path': item['path'],
            'distance': distance,
            'distance_squared': float(squared_l2),
            'similarity': similarity,
            'hash': item['hash'],
        })
        if len(results) >= top_k:
            break

    return {
        'query_file': query_file,
        'query_hash': query_hash,
        'results': results,
    }


def print_faiss_results(result):
    print(f"Query: {result['query_file']}")
    print(f"Matches: {len(result['results'])}")
    for idx, match in enumerate(result['results'], start=1):
        print(
            f"[{idx}] {match['path']} | "
            f"distance={match['distance']:.4f} | "
            f"similarity={match['similarity']:.6f} | "
            f"distance_squared={match['distance_squared']:.4f}"
        )


# ----- Legacy helpers -----

def imgnet_test_inner(i):
    import base64
    filename = f"ILSVRC2012_val_{i + 1:08}.JPEG"
    file_path = "/Volumes/ArcaneNibbl/ILSVRC2012_img_val/" + filename
    photo_hash = base64.b64encode(bytes(compute_hash(file_path))).decode('ascii')
    return (filename, photo_hash)


def imgnet_test():
    import csv
    import multiprocessing

    reference_hashes = {}
    with open('imgnet_hashes.csv', 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            filename, hash_b64 = row
            filename = filename.rsplit('\\', 1)[1]
            reference_hashes[filename] = hash_b64

    p = multiprocessing.Pool()
    results = []
    for i in range(50000):
        results.append(p.apply_async(imgnet_test_inner, [i]))
    with open('imgnettest.txt', 'w') as f:
        for x in results:
            filename, photo_hash = x.get()
            expected_hash = reference_hashes[filename]
            if photo_hash == expected_hash:
                print(f"{filename}: OK", file=f)
            else:
                print(f"{filename}: {expected_hash} {photo_hash}", file=f)
            f.flush()
    p.close()
    p.join()


# ----- CLI -----

def print_usage(argv0: str):
    print('Usage:')
    print(f"  {argv0} image.jpg")
    print(f"  {argv0} image1.jpg image2.jpg")
    print(f"  {argv0} --metric euclidean image1.jpg image2.jpg")
    print(f"  {argv0} --metric manhattan image1.jpg image2.jpg")
    print(f"  {argv0} --faiss-build index.faiss meta.json image1.jpg [image2.jpg ...]")
    print(f"  {argv0} --faiss-add index.faiss meta.json image1.jpg [image2.jpg ...]")
    print(f"  {argv0} --faiss-query index.faiss meta.json query.jpg [top_k]")
    print(f"  {argv0} --faiss-query index.faiss meta.json query.jpg [top_k] --min-similarity 0.95")
    print(f"  {argv0} --faiss-query index.faiss meta.json query.jpg [top_k] --max-distance 15")


def parse_faiss_query_args(args: List[str]):
    if len(args) < 4:
        raise ValueError('Not enough arguments for --faiss-query')

    index_path = args[1]
    meta_path = args[2]
    query_file = args[3]

    top_k = 10
    min_similarity = None
    max_distance = None

    i = 4
    if i < len(args) and not args[i].startswith('--'):
        top_k = int(args[i])
        i += 1

    while i < len(args):
        flag = args[i]
        if flag == '--min-similarity':
            i += 1
            if i >= len(args):
                raise ValueError('--min-similarity requires a value')
            min_similarity = float(args[i])
        elif flag == '--max-distance':
            i += 1
            if i >= len(args):
                raise ValueError('--max-distance requires a value')
            max_distance = float(args[i])
        else:
            raise ValueError(f'Unknown option: {flag}')
        i += 1

    return index_path, meta_path, query_file, top_k, min_similarity, max_distance


def main(argv: List[str]):
    if len(argv) == 2:
        photo_hash = compute_hash(argv[1])
        hash_string = ','.join(str(i) for i in photo_hash)
        print(hash_string)
        return 0

    if len(argv) == 3:
        result = compare_images(argv[1], argv[2])
        print(f"Distance ({result['metric']}): {result['distance']:.4f}")
        print(f"Similarity: {result['similarity']:.6f}")
        return 0

    if len(argv) == 4 and argv[1] == '--metric':
        print(f"Usage: {argv[0]} [--metric euclidean|manhattan] image1 image2")
        return -1

    if len(argv) == 5 and argv[1] == '--metric':
        metric = argv[2]
        result = compare_images(argv[3], argv[4], metric=metric)
        print(f"Distance ({result['metric']}): {result['distance']:.4f}")
        print(f"Similarity: {result['similarity']:.6f}")
        return 0

    if len(argv) >= 5 and argv[1] == '--faiss-build':
        added = build_faiss_index(argv[2], argv[3], argv[4:])
        print(f"Indexed {added} file(s) into {argv[2]}")
        return 0

    if len(argv) >= 5 and argv[1] == '--faiss-add':
        added = add_files_to_faiss(argv[2], argv[3], argv[4:])
        print(f"Added {added} file(s) into {argv[2]}")
        return 0

    if len(argv) >= 5 and argv[1] == '--faiss-query':
        index_path, meta_path, query_file, top_k, min_similarity, max_distance = parse_faiss_query_args(argv[1:])
        result = query_faiss_index(
            index_path=index_path,
            meta_path=meta_path,
            query_file=query_file,
            top_k=top_k,
            min_similarity=min_similarity,
            max_distance=max_distance,
        )
        print_faiss_results(result)
        return 0

    print_usage(argv[0])
    return -1


if __name__ == '__main__':
    sys.exit(main(sys.argv))

# if __name__ == '__main__':
#     imgnet_test()
