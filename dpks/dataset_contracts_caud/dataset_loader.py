from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    CAUD Contracts dataset - items.
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-solutions/CAUDContracts.zip'
    items_path = 'caud_contracts/items/'
    
