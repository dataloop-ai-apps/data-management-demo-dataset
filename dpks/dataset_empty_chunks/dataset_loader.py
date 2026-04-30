import logging

import dtlpy as dl

from base_dataset_loader import BaseDatasetLoader

logger = logging.getLogger('dataloop-dataset-loader')


class DatasetExample(BaseDatasetLoader):
    """
    Reusable empty chunks dataset.

    The dataset is intentionally created empty. Chunks and their embeddings
    are expected to be produced at runtime by the consuming solution's
    preprocessing pipeline, so the standard zip download / extract / item
    upload flow is skipped here.
    """

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress=None):
        logger.info(
            f'Creating empty dataset "{dataset.name}" (id={dataset.id}); no items uploaded.'
        )
        if progress is not None:
            progress.update(progress=100, message='Done', status='Done')
