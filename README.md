# oaphotodna.py

`oaphotodna.py` computes PhotoDNA-like hashes (based on the reversed-engineered version available at [https://github.com/ArcaneNibble/open-alleged-photodna](https://github.com/ArcaneNibble/open-alleged-photodna)) for images, compares two images with normalized similarity scoring, and supports a local FAISS-backed nearest-neighbor index for fast lookup of visually similar images.

This version adds:

- a FAISS local vector index
- persistent on-disk metadata in `meta.json`
- exact L2 nearest-neighbor search
- similarity scores normalized to the same `0..1` scale as direct image comparison
- query-time filtering by minimum similarity or maximum Euclidean distance

## Requirements

- Python 3.8+
- [Pillow](https://pypi.org/project/Pillow/)
- [NumPy](https://pypi.org/project/numpy/) (recommended)
- [faiss-cpu](https://pypi.org/project/faiss-cpu/) for FAISS features

Install dependencies:

```bash
pip install pillow numpy faiss-cpu
```

## What the script does

The script supports three main workflows:

1. Compute the hash of a single image.
2. Compare two images using either Euclidean or Manhattan distance.
3. Build and query a local FAISS index of previously hashed images.

The PhotoDNA-like hash is represented internally as a flat vector of 144 values. FAISS stores these vectors and searches for nearest neighbors using L2 distance.

## Help

Print the top-level help:

```bash
python bin/oaphotodna.py --help
```

The CLI uses traditional, flag-prefixed arguments (for example `--hash`, `--compare`, `--faiss-query`) rather than positional subcommands.

```bash
adulau@blakley:~/git/photodna/bin$ python3 oaphotodna.py
usage: oaphotodna.py [-h] (--hash IMAGE | --compare IMAGE1 IMAGE2 | --faiss-build ARG [ARG ...] | --faiss-add ARG [ARG ...] | --faiss-query ARG [ARG ...]) [--metric {euclidean,manhattan}]
                     [--min-similarity MIN_SIMILARITY] [--max-distance MAX_DISTANCE]

Compute and compare PhotoDNA-like hashes, with optional FAISS local indexing.

options:
  -h, --help            show this help message and exit
  --hash IMAGE          Compute the hash of one image
  --compare IMAGE1 IMAGE2
                        Compare two images
  --faiss-build ARG [ARG ...]
                        Create a new FAISS index: INDEX META IMAGE [IMAGE ...]
  --faiss-add ARG [ARG ...]
                        Append images to an existing FAISS index: INDEX META IMAGE [IMAGE ...]
  --faiss-query ARG [ARG ...]
                        Find closest indexed matches: INDEX META QUERY_IMAGE [TOP_K]
  --metric {euclidean,manhattan}
                        Distance metric for --compare
  --min-similarity MIN_SIMILARITY
                        With --faiss-query, filter results below this similarity threshold [0,1]
  --max-distance MAX_DISTANCE
                        With --faiss-query, filter results above this Euclidean distance
```

## Basic usage

### 1) Hash a single image

```bash
python bin/oaphotodna.py --hash image.jpg
```

Output:

```text
73,71,74,32,...
```

### 2) Compare two images

Default metric is Euclidean:

```bash
python bin/oaphotodna.py --compare image1.jpg image2.jpg
```

Use Manhattan distance instead:

```bash
python bin/oaphotodna.py --compare image1.jpg image2.jpg --metric manhattan
```

Example output:

```text
Distance (euclidean): 3.7417
Similarity: 0.998779
```

## Similarity scale

The script reports a normalized similarity value between `0` and `1`.

- `1.0` means identical hashes
- values close to `1.0` mean very similar hashes
- values closer to `0.0` mean more distant hashes

For Euclidean distance, similarity is derived from the maximum possible distance for a 144-dimensional hash with values in the range `0..255`:

```text
similarity = 1 - (euclidean_distance / max_possible_distance)
```

The FAISS query path uses the same normalization so that the `similarity` reported by `--faiss-query` is directly comparable to the `Similarity:` line from `--compare`.

## FAISS local database

### Files used

The local database consists of two files:

- `index.faiss` — the FAISS vector index
- `meta.json` — sidecar metadata used to map FAISS IDs back to files and hashes

### What `meta.json` contains

`meta.json` stores information that FAISS does not store for you in an application-friendly way:

- `dimension` — vector length, normally `144`
- `metric` — stored metric type
- `next_id` — next numeric ID to assign
- `items` — indexed records

Each item in `items` contains:

- `id` — numeric FAISS ID
- `path` — canonicalized file path
- `hash` — stored 144-element hash
- `extra` — optional metadata placeholder

## Build an index

Create a new index from a set of images:

```bash
python bin/oaphotodna.py --faiss-build index.faiss meta.json img1.jpg img2.jpg img3.jpg
```

Expected output:

```text
Indexed 3 file(s) into index.faiss
```

## Add images to an existing index

Append more images later:

```bash
python bin/oaphotodna.py --faiss-add index.faiss meta.json img4.jpg img5.jpg
```

Expected output:

```text
Added 2 file(s) into index.faiss
```

## Query the index

Search for the closest matches to a query image:

```bash
python bin/oaphotodna.py --faiss-query index.faiss meta.json query.jpg
```

Specify the number of results to return:

```bash
python bin/oaphotodna.py --faiss-query index.faiss meta.json query.jpg 20
```

Example output:

```text
Query: query.jpg
Results: 3

[1] /data/images/img2.jpg
    id=17
    distance=3.7417
    similarity=0.998779
    distance_squared=14.0000

[2] /data/images/img7.jpg
    id=42
    distance=5.2915
    similarity=0.998273
    distance_squared=28.0000
```

### Filter query results by similarity

Only return matches at or above a similarity threshold:

```bash
python bin/oaphotodna.py --faiss-query index.faiss meta.json query.jpg 20 --min-similarity 0.95
```

### Filter query results by Euclidean distance

Only return matches at or below a maximum Euclidean distance:

```bash
python bin/oaphotodna.py --faiss-query index.faiss meta.json query.jpg 20 --max-distance 12
```

### Combine both filters

```bash
python bin/oaphotodna.py --faiss-query index.faiss meta.json query.jpg 20 --min-similarity 0.98 --max-distance 8
```

## FAISS distance notes

FAISS returns **squared L2 distance** internally.

The script converts that into:

- `distance_squared` — raw FAISS value
- `distance` — Euclidean distance (`sqrt(distance_squared)`)
- `similarity` — normalized `0..1` score derived from Euclidean distance
