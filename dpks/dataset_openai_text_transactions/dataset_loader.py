from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    OpenAI Transactions Text dataset - items + feature set (text-embeddings-3).
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-solutions/CompanyDocumentsChunks.zip'
    items_path = 'CompanyDocuments Chunks/items/'

    feature_sets = [
        dict(
            name='openai-text-embeddings-3l',
            type='text-embeddings',
            vectors_path='CompanyDocuments Chunks/vectors/openai-text-embeddings-3l.json',
            model_dpk_name='text-embeddings-3',
            model_component_name='openai-text-embeddings-3l',
        )
    ]
