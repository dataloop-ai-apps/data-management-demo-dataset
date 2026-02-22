from base_dataset_loader import BaseDatasetLoader


class DatasetExample(BaseDatasetLoader):
    """
    Vehicle dashcam image dataset – raw items, no annotations.
    No annotations: the use case is pre-annotation quality triage.
    """

    zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-kaggle/vehicle_dashcam_image_dataset.zip'
    items_path = 'vehicle_dashcam_image_dataset/items/'
