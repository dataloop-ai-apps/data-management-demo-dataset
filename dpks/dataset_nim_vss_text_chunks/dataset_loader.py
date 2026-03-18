from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    NIM VSS text chunks dataset - items + annotations + feature set (nim-llama-3-2-nemoretriever-1b-vlm-embed-v1).
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-solutions/VSSChunks.zip'
    items_path = 'VSSChunks/items/'

    feature_sets = [
        dict(
            name='nim-llama-3-2-nemoretriever-1b-vlm-embed-v1',
            type='text-embeddings',
            vectors_path='VSSChunks/vectors/nim-llama-3-2-nemoretriever-1b-vlm-embed-v1.json',
            model_dpk_name='nim-llama-3-2-nemoretriever-1b-vlm-embed-v1',
            model_component_name='nim-llama-3-2-nemoretriever-1b-vlm-embed-v1',
        )
    ]
