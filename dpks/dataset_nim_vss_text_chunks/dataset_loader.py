from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    NIM VSS text chunks dataset - items + annotations + feature set (nim-nemotron-3-5-embed-1b).
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/vss-solution/VSSChunks_nemotron_3_5_embed_1b.zip'
    items_path = 'VSSChunks/items/'

    feature_sets = [
        dict(
            name='nim-nemotron-3-5-embed-1b',
            type='text-embeddings',
            vectors_path='VSSChunks/vectors/nim-nemotron-3-5-embed-1b.json',
            model_dpk_name='nim-nemotron-3-5-embed-1b',
            model_component_name='nim-nemotron-3-5-embed-1b',
        )
    ]
