from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import pandas as pd
import dtlpy as dl
import logging
import requests
import zipfile
import json
import tqdm
import os

logger = logging.getLogger('dataloop-example-dataset-rag')


class DatasetExample(dl.BaseServiceRunner):
    """
    A class to handle upload of example dataset to Dataloop platform.
    """

    def __init__(self):
        """
        Initialize the dataset downloader.
        """
        self.dir = os.getcwd()
        self.embedding_model_dpk_name = None
        self.embedding_model_component_name = None
        self.feature_set_name = None
        self.feature_set_type = None

        logger.info('Dataset loader initialized.')

    def upload_dataset(self, dataset: dl.Dataset, source: str, progress=None):
        """
        Uploads the dataset to Dataloop platform, including items, annotations and feature vectors.

        :param dataset: The Dataloop dataset object where the data will be uploaded.
        """
        if source == 'Dataloop':
            self.embedding_model_dpk_name = 'text-embeddings-3'
            self.embedding_model_component_name = 'openai-text-embeddings-3l'
            self.feature_set_name = 'openai-text-embeddings-3l'
            self.feature_set_type = 'text-embeddings'
            zip_url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-rag/export.zip'
        elif source == 'NIM':
            self.embedding_model_dpk_name = 'nim-llama-3-2-nemoretriever-300m-embed-v2'
            self.embedding_model_component_name = 'nim-llama-3-2-nemoretriever-300m-embed-v2'
            self.feature_set_name = 'nim-llama-3-2-nemoretriever-300m-embed-v2'
            self.feature_set_type = 'text-embeddings'
            zip_url = 'TODO/data.zip'
        else:
            raise ValueError(f'Invalid source: {source}')

        progress.update(progress=0,
                        message='Creating dataset...',
                        status='Creating dataset...')

        logger.info('Uploading dataset...')
        self.extract_zip(zip_url)
        extracted_dir = os.path.basename(zip_url).replace('.zip', '')
        local_path = os.path.join(self.dir, extracted_dir, 'items')

        progress_tracker = {'last_progress': 0}

        def progress_callback_all(progress_class, progress, context):
            new_progress = progress // 2
            if new_progress > progress_tracker['last_progress'] and new_progress % 10 == 0:
                progress_tracker['last_progress'] = new_progress
                if progress_class is not None:
                    progress_class.update(progress=new_progress,
                                          message=f'Uploading items and annotations ...',
                                          status=f'Uploading items and annotations ...')

        progress_callback = partial(progress_callback_all, progress)

        dl.client_api.add_callback(func=progress_callback, event=dl.CallbackEvent.ITEMS_UPLOAD)

        to_upload = list()
        for filename in os.listdir(local_path):
            file_path = os.path.join(local_path, filename)
            if os.path.isfile(file_path):
                to_upload.append(dict(local_path=file_path,
                                      item_metadata={'system': {'document': {'source': filename}}}))

        dataset.items.upload(pd.DataFrame(to_upload))

        # Handle feature set
        feature_set = self.ensure_feature_set(dataset)

        # Upload features
        vectors_file = os.path.join(self.dir, extracted_dir, 'vectors', 'vectors.json')
        with open(vectors_file, 'r') as f:
            vectors = json.load(f)

        total_tasks = len(vectors)
        with tqdm.tqdm(total=total_tasks, desc='Uploading features') as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.create_feature, key, value, dataset, feature_set) for key, value in
                           vectors.items()]
                task_done = 0
                for future in as_completed(futures):
                    pbar.update(1)
                    task_done += 1
                    new_progress = (task_done * 50) // total_tasks + 50
                    if new_progress > progress_tracker['last_progress'] and new_progress % 10 == 0:
                        progress_tracker['last_progress'] = new_progress
                        if progress is not None:
                            progress.update(progress=new_progress,
                                            message=f'Uploading feature set ...',
                                            status=f'Uploading feature set ...')

    def get_model(self, project: dl.Project):
        dpk = dl.dpks.get(dpk_name=self.embedding_model_dpk_name)
        try:
            app: dl.App = project.apps.get(app_name=dpk.display_name)
        except dl.exceptions.NotFound:
            app: dl.App = project.apps.install(dpk=dpk)

        filters = dl.Filters(resource=dl.FiltersResource.MODELS, field='app.id', values=app.id)
        filters.add(field='app.componentName', values=self.embedding_model_component_name)
        models = list(project.models.list(filters=filters).all())
        if len(models) == 0:
            model: dl.Model = app.models.create(
                model_name=self.embedding_model_component_name,
                dpk_model_name=self.embedding_model_name,
                output_type=self.feature_set_type,
            )
        elif len(models) == 1:
            model: dl.Model = models[0]
        else:
            raise ValueError(f'Multiple models found for {self.embedding_model_component_name}') 
        return model

    def ensure_feature_set(self, dataset):
        """
        Ensures that the feature set exists or creates a new one if not found.

        :param dataset: The dataset where the feature set is to be managed.
        """
        
        project: dl.Project = dataset.project
        model = self.get_model(project=project)
        try:
            feature_set = project.feature_sets.get(feature_set_name=self.feature_set_name)
            logger.info(f'Feature Set found! Name: {feature_set.name}, ID: {feature_set.id}')
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found, creating...')
            feature_set = project.feature_sets.create(
                name=self.feature_set_name,
                entity_type=dl.FeatureEntityType.ITEM,
                project_id=project.id,
                set_type=self.feature_set_type,
                size=model.configuration['embeddings_size'],
                model_id=model.id
            )
        return feature_set

    @staticmethod
    def create_feature(key, value, dataset, feature_set):
        """
        Creates a feature for a given item.

        :param key: The key identifying the item.
        :param value: The feature value to be added.
        :param dataset: The dataset containing the item.
        :param feature_set: The feature set to which the feature will be added.
        """
        item = dataset.items.get(filepath='/' + key)
        feature_set.features.create(entity=item, value=value)

    def extract_zip(self, zip_url):
        """
        Extracts the zip file.

        :param zip_url: The url to the zip file.
        """

        logger.info('Downloading zip file...')
        zip_dir = os.path.join(self.dir, os.path.basename(zip_url))
        # Download the zip file
        response = requests.get(zip_url)
        if response.status_code == 200:
            with open(zip_dir, 'wb') as f:
                f.write(response.content)
        else:
            logger.error(f'Failed to download the file. Status code: {response.status_code}')
            return

        # Extract the zip file
        with zipfile.ZipFile(zip_dir, 'r') as zip_ref:
            zip_ref.extractall(self.dir)
        logger.info('Zip file downloaded and extracted.')
