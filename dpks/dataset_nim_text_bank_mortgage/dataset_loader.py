from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    NIM Bank Mortgage Text dataset - items + feature set (nim-llama-3-2-nemoretriever-300m-embed-v2).
    """

    zip_url = 'TODO/data.zip'
    items_path = 'Truth in Lending Act Chunks/items/'

    feature_sets = [
        dict(
            name='nim-llama-3-2-nemoretriever-300m-embed-v2',
            type='text-embeddings',
            vectors_path='Truth in Lending Act Chunks/vectors/nim-llama-3-2-nemoretriever-300m-embed-v2.json',
            model_dpk_name='nim-llama-3-2-nemoretriever-300m-embed-v2',
            model_component_name='nim-llama-3-2-nemoretriever-300m-embed-v2',
        )
    ]
