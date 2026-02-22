from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    Retail product dataset – items + two feature sets (CLIP + ResNet).
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-kaggle/retail_products.zip'
    items_path = 'retail_products/items/'

    feature_sets = [
        dict(
            name='openai-clip',
            type='clip',
            size=512,
            vectors_path='retail_products/vectors/openai-clip.json',
        ),
        dict(
            name='pretrained-resnet',
            type='resnet',
            size=2048,
            vectors_path='retail_products/vectors/pretrained-resnet.json',
        ),
    ]
