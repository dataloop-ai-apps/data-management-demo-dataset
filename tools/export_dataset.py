"""
Export a DDOE dataset to a local zip, ready for use with BaseDatasetLoader.

Edit the configuration section below, then run:
    python tools/export_dataset.py
"""

import os
import json
import shutil
import dtlpy as dl

# ──────────────────────────────────────────────────────────────
#  USER CONFIGURATION
# ──────────────────────────────────────────────────────────────

PROJECT_ID = 'your_project_id'
DATASET_NAME = 'your_dataset_name'
# Alternatively, set DATASET_ID directly (takes precedence):
DATASET_ID = None

BASE_DIR = None  # e.g. 'VSS Text Chunks'  (None = use dataset name)

EXTRACT_ANNOTATIONS = True
EXTRACT_EMBEDDINGS = True

# ──────────────────────────────────────────────────────────────

dl.setenv('prod')


def get_dataset() -> dl.Dataset:
    if DATASET_ID:
        return dl.datasets.get(dataset_id=DATASET_ID)
    project = dl.projects.get(project_id=PROJECT_ID)
    return project.datasets.get(dataset_name=DATASET_NAME)


def download_items(dataset: dl.Dataset, base_dir: str):
    annotation_options = []
    if EXTRACT_ANNOTATIONS:
        annotation_options.append(dl.ViewAnnotationOptions.JSON)

    dataset.download(
        local_path=base_dir,
        annotation_options=annotation_options if annotation_options else None,
    )

    items_dir = os.path.join(base_dir, 'items')
    if not os.path.isdir(items_dir):
        print('[warning] No items downloaded.')

    if EXTRACT_ANNOTATIONS:
        json_dir = os.path.join(base_dir, 'json')
        annotations_dir = os.path.join(base_dir, 'annotations')
        if os.path.isdir(json_dir):
            os.rename(json_dir, annotations_dir)
            print(f'Renamed {json_dir} -> {annotations_dir}')


def extract_embeddings(dataset: dl.Dataset, base_dir: str):
    vectors_dir = os.path.join(base_dir, 'vectors')
    os.makedirs(vectors_dir, exist_ok=True)

    for fs in dataset.feature_sets.list().all():
        print(f'Extracting feature set: {fs.name} (size={fs.size}, id={fs.id})')
        vectors = {}
        for feature in fs.features.list().all():
            item = dataset.items.get(item_id=feature.entity_id)
            vectors[item.filename] = feature.value
            print(f'  {item.filename} -> vector[{len(feature.value)}]')

        out_path = os.path.join(vectors_dir, f'{fs.name}.json')
        with open(out_path, 'w') as f:
            json.dump(vectors, f)
        print(f'Saved {len(vectors)} vectors to {out_path}')


def create_zip(base_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = shutil.make_archive(
        os.path.join(output_dir, os.path.basename(base_dir)),
        'zip', '.', base_dir,
    )
    print(f'Created {zip_path}')


def main():
    dataset = get_dataset()

    base_dir = BASE_DIR or dataset.name
    if os.path.isdir(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    print(f'Exporting dataset "{dataset.name}" -> {base_dir}/')

    download_items(dataset, base_dir)

    if EXTRACT_EMBEDDINGS:
        extract_embeddings(dataset, base_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    create_zip(base_dir, output_dir)

    print('\n── Paths for dataset_loader.py ──')
    print(f"  items_path = '{base_dir}/items/'")
    if EXTRACT_ANNOTATIONS and os.path.isdir(os.path.join(base_dir, 'annotations')):
        print(f"  annotations_path = '{base_dir}/annotations/'")
    if EXTRACT_EMBEDDINGS:
        vectors_dir = os.path.join(base_dir, 'vectors')
        if os.path.isdir(vectors_dir):
            for name in os.listdir(vectors_dir):
                print(f"  vectors_path = '{base_dir}/vectors/{name}'")

    shutil.rmtree(base_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
