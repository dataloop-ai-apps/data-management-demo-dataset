import os
from functools import partial
import dtlpy as dl
from base_dataset_loader import BaseDatasetLoader


class DatasetLoader(BaseDatasetLoader):
    """
    Medical records dataset – PDF assets + text chunks.
    Zip structure:
        medical_dataset/assets/   → uploaded to /
        medical_dataset/chunks/   → uploaded to /chunks
    """

    zip_url = "https://storage.googleapis.com/model-mgmt-snapshots/datasets-medical/medical_dataset.zip"
    assets_path = 'medical_dataset/assets/'
    chunks_path = 'medical_dataset/chunks/'

    # Not used directly, but keeps base happy for the items step
    items_path = 'medical_dataset/assets/'

    def _upload_items(self, dataset: dl.Dataset, progress_tracker, with_annotations: bool):
        callback = partial(self._items_progress_callback, progress_tracker)
        dl.client_api.add_callback(func=callback, event=dl.CallbackEvent.ITEMS_UPLOAD)

        assets_dir = os.path.join(self.dir, self.assets_path)
        chunks_dir = os.path.join(self.dir, self.chunks_path)

        # Fall back to local dirs next to this file (original behavior)
        if not os.path.isdir(assets_dir):
            assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        if not os.path.isdir(chunks_dir):
            chunks_dir = os.path.join(os.path.dirname(__file__), 'chunks')

        dataset.items.upload(local_path=assets_dir)
        dataset.items.upload(local_path=chunks_dir, remote_path='/chunks')
