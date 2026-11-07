"""Town Neck geometry, split validation, and Svalbard checkpoint transfer."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from Source.beach_dataset import BeachCellDataset as BaseDataset, load_cell_context_array, _augment_spatial
from Source.bs_splits import file_sha256
from Source.bs_town_neck_splits import validate_splits
from Source.ensemble import load_model_artifact
from Source.inference import load_model_for_inference
from Source.training import predict


def current_split_provenance(directory):
    directory = Path(directory)
    p = json.loads((directory / 'bs_split_provenance.json').read_text())
    if p['policy'] != 'town_neck_stratified_sample_spatial_blocks_v1':
        raise ValueError('Not a Town Neck split protocol')
    for name, expected in p['csv_sha256'].items():
        filename = 'bs_dataset_splits.csv' if name == 'all' else f'bs_{name}_split.csv'
        if file_sha256(directory / filename) != expected:
            raise ValueError('Split files changed; regenerate splits and retrain')
    raw = directory.parent.parent / 'Raw' / 'bs_town_neck'
    for filename,key in [('Descriptor_bs_town_neck.csv','raw_descriptor_sha256'),
                         ('bs_town_neck_grid_metadata.json','raw_grid_metadata_sha256')]:
        if file_sha256(raw / filename) != p[key]:
            raise ValueError('Raw descriptor/grid differs from split provenance')
    return p


def validate_frames(train, validation, test, raw_root):
    raw_root = Path(raw_root)
    grid = json.loads((raw_root / 'bs_town_neck_grid_metadata.json').read_text())
    source = pd.read_csv(raw_root / 'Descriptor_bs_town_neck.csv', dtype={'date':str})
    frames = {}
    for name,frame in zip(['train','validation','test'],[train,validation,test]):
        frame = frame.copy()
        frame['cell_id'] = frame.cell_id.astype(int)
        frame['class'] = frame['class'].astype(int)
        frames[name] = frame
    return validate_splits(frames, source, grid['grid_rows'], grid['grid_columns'], 3,
                           {k:len(v) for k,v in frames.items()})


class TownNeckDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        grid = json.loads((self.annotated_root.parent / 'bs_town_neck_grid_metadata.json').read_text())
        self.grid_rows, self.grid_columns = grid['grid_rows'], grid['grid_columns']
        if self.feature_suffixes != ('r','g','b') or self.context_radius not in range(4):
            raise ValueError('Town Neck supports RGB only and radii 0-3')

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        array = load_cell_context_array(self.annotated_root, str(row['date']), str(row['cell_id']),
                    self.feature_suffixes, self.context_radius, self.grid_rows, self.grid_columns)
        x = torch.from_numpy(array)
        if self.mean is None or self.std is None:
            raise ValueError('Supply training-only normalization statistics')
        # Fill before interpolation so NaNs cannot spread into neighboring valid pixels.
        x = torch.where(torch.isfinite(x), x, self.mean)
        if self.augment:
            x = _augment_spatial(x)
        if self.image_size is not None:
            x = F.interpolate(x.unsqueeze(0), size=(self.image_size,self.image_size),
                              mode='bilinear', align_corners=False).squeeze(0)
        x = (x-self.mean)/self.std.clamp_min(1e-6)
        return x, torch.tensor(self.class_to_idx[str(row['class'])], dtype=torch.long)


def transfer_source(model_type, result_root):
    artifact = load_model_artifact(model_type, Path(result_root) / f'bs_{model_type}' / 'RGB_only')
    if artifact.metadata['feature_suffixes'] != ['r','g','b']:
        raise ValueError('Transfer requires the matching RGB-only Svalbard checkpoint')
    if artifact.metadata['model_type'] != model_type or artifact.metadata['num_classes'] != 10:
        raise ValueError('Expected the matching ten-class Svalbard architecture')
    return artifact


def initialization_metadata(artifact):
    return {'type':'svalbard_rgb_checkpoint', 'model_type':artifact.name,
            'checkpoint_sha256':file_sha256(artifact.checkpoint_path),
            'source_context_radius':artifact.metadata['context_radius'],
            'source_best_config':artifact.metadata['best_config'],
            'output_layer':'fresh six-class linear layer', 'trainable_layers':'all'}


def initialize_from_svalbard(model, artifact):
    if file_sha256(artifact.checkpoint_path) != artifact.metadata['checkpoint_sha256']:
        raise ValueError('Transfer checkpoint changed during grid search')
    state = torch.load(artifact.checkpoint_path, map_location='cpu', weights_only=True)
    head = {'small_cnn':'classifier.5', 'convnext_tiny':'classifier.2',
            'vit_tiny':'head', 'swin_tiny':'head.fc', 'efficientnetv2_s':'classifier'}[artifact.name]
    excluded = {f'{head}.weight', f'{head}.bias'}
    target = model.state_dict()
    # timm versions can use either head.fc or head for Swin's final linear layer.
    if artifact.name == 'swin_tiny' and f'{head}.weight' not in target:
        excluded = {'head.weight','head.bias'}
    if not excluded.issubset(state) or not excluded.issubset(target):
        raise ValueError('Unexpected output layer; refusing partial backbone transfer')
    for key in excluded:
        if target[key].shape[0] != 6 or state[key].shape[0] != 10:
            raise ValueError('Expected six-class target and ten-class source heads')
    incompatible = model.load_state_dict({k:v for k,v in state.items() if k not in excluded}, strict=False)
    if set(incompatible.missing_keys) != excluded or incompatible.unexpected_keys:
        raise ValueError(f'Incompatible source architecture: {incompatible}')
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def predict_artifact_on_split(artifact, split_df, cell_data_root, device, batch_size,
                              num_workers, show_progress=False):
    m = artifact.metadata
    grid = json.loads((Path(cell_data_root).parent / 'bs_town_neck_grid_metadata.json').read_text())
    if m.get('grid_rows') != grid['grid_rows'] or m.get('grid_columns') != grid['grid_columns']:
        raise ValueError('Town Neck inference geometry does not match the checkpoint')
    dataset = TownNeckDataset(split_df, cell_data_root, m['class_to_idx'], image_size=m['image_size'],
                 mean=m['channel_mean'], std=m['channel_std'], feature_suffixes=m['feature_suffixes'],
                 context_radius=m['context_radius'])
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    model = load_model_for_inference(artifact.checkpoint_path, m, device)
    truth, _, probabilities = predict(model, loader, device, show_progress=show_progress)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return truth, probabilities
