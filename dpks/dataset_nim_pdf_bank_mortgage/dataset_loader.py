from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    NIM Bank Mortgage PDF dataset - items.
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-solutions/TruthInLendingAct.zip'
    items_path = 'Truth in Lending Act/items/'
