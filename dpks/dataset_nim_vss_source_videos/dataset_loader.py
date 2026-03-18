from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    NIM VSS source videos dataset - items only (no feature set).
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-solutions/VSSVideos.zip'
    items_path = 'VSSVideos/items/'
