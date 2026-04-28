from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    CAUD Contracts Chunks dataset - items + feature set (clip-model-pretrained).
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-solutions/caud_contracts.zip'
    items_path = 'caud_contracts/chunks/'

    feature_sets = [
        dict(
            name='clip-model-pretrained',
            type='text-embeddings',
            vectors_path='caud_contracts/vectors/openai-clip.json',
            model_dpk_name='clip-model-pretrained',
            model_component_name='clip-model-pretrained',
        )
    ]
