import logging
import os

import dtlpy as dl

logger = logging.getLogger('medical-dataset-loader')

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), 'chunks')


class DatasetLoader(dl.BaseServiceRunner):

    def __init__(self):
        """
        Initialize the dataset downloader.
        """

        logger.info('Dataset loader initialized.')

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress=None):
        if progress is not None:
            progress.update(progress=0,
                            message='Uploading medical records...',
                            status='Uploading medical records...')

        logger.info(f'Uploading items from {ASSETS_DIR}')
        dataset.items.upload(local_path=ASSETS_DIR)

        if progress is not None:
            progress.update(progress=50,
                            message='Uploading chunks...',
                            status='Uploading chunks...')

        logger.info(f'Uploading chunks from {CHUNKS_DIR}')
        dataset.items.upload(local_path=CHUNKS_DIR, remote_path='/chunks')

        if progress is not None:
            progress.update(progress=100,
                            message='Upload complete.',
                            status='Upload complete.')
        logger.info('Dataset upload complete.')
