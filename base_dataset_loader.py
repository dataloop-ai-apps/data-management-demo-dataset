import logging
import os
import requests
import zipfile
import json
import dtlpy as dl
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import tqdm

logger = logging.getLogger('dataloop-dataset-loader')


class BaseDatasetLoader(dl.BaseServiceRunner):
    """
    Reusable base for Dataloop dataset DPKs.

    Subclasses configure the upload by setting class-level attributes:
        zip_url:            URL to the zipped export
        items_path:         Relative path to items inside the extracted zip
        annotations_path:   Relative path to annotations (None to skip)
        labels:             List of label strings to add to the ontology (None to skip)

    Embeddings – pick ONE style:
        feature_sets:       List of dicts for multiple feature sets, each with keys:
                                name, type, size, vectors_path
                            Optional model-linking keys (per feature set):
                                model_dpk_name, model_name, model_component_name
                            When model keys are present, size and type are read from
                            the model, so they can be omitted from the dict.
        OR for a single feature set use the shorthand attributes:
            vectors_path, feature_set_name, feature_set_type, feature_set_size
    """

    zip_url: str = None
    items_path: str = 'export/items/'
    annotations_path: str = None
    labels: list = None

    feature_sets: list = None
    vectors_path: str = None
    feature_set_name: str = None
    feature_set_type: str = None
    feature_set_size: int = None

    def __init__(self):
        self.dir = os.getcwd()
        logger.info('Dataset loader initialized.')

    # ------------------------------------------------------------------
    # Main entry point – called by the DPK
    # ------------------------------------------------------------------
    def upload_dataset(self, dataset: dl.Dataset, source: str, progress=None):
        if progress is not None:
            progress.update(progress=0,
                            message='Creating dataset...',
                            status='Creating dataset...')

        logger.info('Uploading dataset...')
        self.extract_zip(self.zip_url)

        has_annotations = self.annotations_path is not None
        fs_configs = self._get_feature_set_configs()

        steps = ['items']
        if has_annotations:
            steps.append('annotations')
        for i, _ in enumerate(fs_configs):
            steps.append(f'embeddings_{i}')

        progress_tracker = _ProgressTracker(progress, steps)

        self._upload_items(dataset, progress_tracker, has_annotations)

        if self.labels:
            self._setup_ontology(dataset)

        for i, fs_cfg in enumerate(fs_configs):
            self._upload_embeddings(dataset, progress_tracker, fs_cfg, step_name=f'embeddings_{i}')

        if progress is not None:
            progress.update(progress=100,
                            message='Done',
                            status='Done')

    # ------------------------------------------------------------------
    # Item upload – override in subclass if you need custom per-item logic
    # ------------------------------------------------------------------
    def _upload_items(self, dataset: dl.Dataset, progress_tracker, with_annotations: bool):
        local_path = os.path.join(self.dir, self.items_path)

        callback = partial(
            self._items_progress_callback,
            progress_tracker
        )
        dl.client_api.add_callback(func=callback, event=dl.CallbackEvent.ITEMS_UPLOAD)

        if with_annotations:
            json_path = os.path.join(self.dir, self.annotations_path)
            dataset.items.upload(local_path=local_path,
                                 local_annotations_path=json_path,
                                 item_metadata=dl.ExportMetadata.FROM_JSON)
        else:
            dataset.items.upload(local_path=local_path)

    # ------------------------------------------------------------------
    # Ontology / labels
    # ------------------------------------------------------------------
    def _setup_ontology(self, dataset: dl.Dataset):
        recipe = dataset.recipes.list()[0]
        ontology = recipe.ontologies.list()[0]
        ontology.add_labels(label_list=self.labels)
        recipe.update()

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def _get_feature_set_configs(self) -> list:
        """
        Returns a normalised list of feature-set config dicts.
        Supports both the multi-set ``feature_sets`` list and the single-set
        shorthand attributes.
        """
        if self.feature_sets:
            return list(self.feature_sets)
        if self.vectors_path:
            return [dict(
                name=self.feature_set_name,
                type=self.feature_set_type,
                size=self.feature_set_size,
                vectors_path=self.vectors_path,
            )]
        return []

    def _upload_embeddings(self, dataset: dl.Dataset, progress_tracker,
                           fs_cfg: dict, step_name: str):
        model = None
        if 'model_dpk_name' in fs_cfg:
            model = self._get_or_create_model(dataset.project, fs_cfg)
        feature_set = self._ensure_feature_set(dataset, fs_cfg, model=model)

        vectors_file = os.path.join(self.dir, fs_cfg['vectors_path'])
        with open(vectors_file, 'r') as f:
            vectors = json.load(f)

        total = len(vectors)
        with tqdm.tqdm(total=total, desc=f"Uploading features ({fs_cfg['name']})") as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(self._create_feature, key, value, dataset, feature_set)
                    for key, value in vectors.items()
                ]
                for i, future in enumerate(as_completed(futures), 1):
                    pbar.update(1)
                    progress_tracker.update_step(step_name, i, total)

    @staticmethod
    def _get_or_create_model(project: dl.Project, fs_cfg: dict) -> dl.Model:
        dpk = dl.dpks.get(dpk_name=fs_cfg['model_dpk_name'])
        try:
            app = project.apps.get(app_name=dpk.display_name)
        except dl.exceptions.NotFound:
            app = project.apps.install(dpk=dpk)

        model_component_name = fs_cfg['model_component_name']
        try:
            model = project.models.get(model_name=model_component_name)
        except dl.exceptions.NotFound:
            model = app.models.create(
                model_name=model_component_name,
                dpk_model_name=dpk.name,
                output_type='embedding',
            )
        return model

    @staticmethod
    def _ensure_feature_set(dataset: dl.Dataset, fs_cfg: dict,
                            model: dl.Model = None):
        fs_name = fs_cfg['name']
        try:
            feature_set = dataset.project.feature_sets.get(
                feature_set_name=fs_name
            )
            logger.info(f'Feature Set found! Name: {feature_set.name}, ID: {feature_set.id}')
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found, creating...')
            create_kwargs = dict(
                name=fs_name,
                entity_type=dl.FeatureEntityType.ITEM,
                project_id=dataset.project.id,
                set_type=fs_cfg.get('type', 'clip'),
                size=fs_cfg.get('size'),
            )
            if model is not None:
                create_kwargs['size'] = model.configuration.get(
                    'embeddings_size', create_kwargs['size']
                )
                create_kwargs['model_id'] = model.id
            feature_set = dataset.project.feature_sets.create(**create_kwargs)
        return feature_set

    @staticmethod
    def _create_feature(key, value, dataset, feature_set):
        filepath = key if key.startswith('/') else '/' + key
        item = dataset.items.get(filepath=filepath)
        feature_set.features.create(entity=item, value=value)

    # ------------------------------------------------------------------
    # Zip download / extract
    # ------------------------------------------------------------------
    def extract_zip(self, zip_source):
        """Accepts a URL (http/https) or a local file path."""
        if zip_source.startswith(('http://', 'https://')):
            logger.info('Downloading zip file...')
            zip_path = os.path.join(self.dir, 'export.zip')
            response = requests.get(zip_source)
            if response.status_code != 200:
                raise RuntimeError(f'Failed to download zip. Status code: {response.status_code}')
            with open(zip_path, 'wb') as f:
                f.write(response.content)
        else:
            zip_path = zip_source
            if not os.path.isabs(zip_path):
                zip_path = os.path.join(self.dir, zip_path)
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f'Local zip not found: {zip_path}')
            logger.info(f'Using local zip: {zip_path}')

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dir)
        logger.info('Zip file extracted.')

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _items_progress_callback(progress_tracker, progress, context):
        progress_tracker.update_step('items', progress, 100)


class _ProgressTracker:
    """
    Splits 0-100 progress evenly across N named steps and deduplicates
    updates so only multiples of a given granularity are reported.
    """

    def __init__(self, progress_obj, steps: list, granularity: int = 5):
        self._progress = progress_obj
        self._steps = steps
        self._granularity = granularity
        self._last_reported = -1

        n = len(steps)
        self._ranges = {}
        for i, step in enumerate(steps):
            start = int(i * 100 / n)
            end = int((i + 1) * 100 / n)
            self._ranges[step] = (start, end)

        self._messages = {
            'items': 'Uploading items ...',
            'annotations': 'Uploading items and annotations ...',
        }
        for step in steps:
            if step.startswith('embeddings'):
                self._messages[step] = 'Uploading feature set ...'

    def update_step(self, step: str, current: int, total: int):
        if self._progress is None:
            return
        start, end = self._ranges[step]
        pct = start + int((current / max(total, 1)) * (end - start))
        pct = min(pct, end)
        if pct > self._last_reported and pct % self._granularity == 0:
            self._last_reported = pct
            msg = self._messages.get(step, f'Processing {step} ...')
            self._progress.update(progress=pct, message=msg, status=msg)
