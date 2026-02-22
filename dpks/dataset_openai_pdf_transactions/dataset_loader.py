from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    OpenAI Transactions PDF dataset - items.
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-solutions/CompanyDocuments.zip'
    items_path = 'CompanyDocuments/items/'
