from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    CAUD Contracts dataset - items.
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-caud-contracts/caud_contracts_pdf.zip'
    items_path = 'caud_contracts_pdf/items/'
