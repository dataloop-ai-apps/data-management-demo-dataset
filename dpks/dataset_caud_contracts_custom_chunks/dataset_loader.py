import logging

import dtlpy as dl

from base_dataset_loader import BaseDatasetLoader

logger = logging.getLogger('dataloop-dataset-loader')


class DatasetExample(BaseDatasetLoader):
    """
    Custom CAUD Contracts Chunks dataset.

    The chunks dataset is intentionally created empty. Chunks and their CLIP
    embeddings are produced at runtime by the preprocess pipeline, so the
    standard zip download / extract / item upload flow is skipped here.
    """

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress=None):
        logger.info(
            f'Creating empty dataset "{dataset.name}" (id={dataset.id}); no items uploaded.'
        )
        if progress is not None:
            progress.update(progress=100, message='Done', status='Done')
