import logging

import dtlpy as dl

from base_dataset_loader import BaseDatasetLoader

logger = logging.getLogger('dataloop-dataset-loader')


class DatasetExample(BaseDatasetLoader):
    """
    Reusable empty source dataset.

    The dataset is intentionally created empty. Items are expected to be
    ingested at runtime by the consuming solution (e.g. via a pipeline
    trigger), so the standard zip download / extract / item upload flow is
    skipped here.
    """

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress=None):
        logger.info(
            f'Creating empty dataset "{dataset.name}" (id={dataset.id}); no items uploaded.'
        )
        if progress is not None:
            progress.update(progress=100, message='Done', status='Done')
