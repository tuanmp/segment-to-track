import logging

import lightning as L
from torch.utils.data import DataLoader, default_collate

from .point_classifier_dataset import PointCloudDataset


def predict_collate_fn(batch):

    to_default_collate = [
        batch[i][:-2] for i in range(len(batch))
    ]

    events = [batch[i][-1] for i in range(len(batch))]
    event_idxs = [batch[i][-2] for i in range(len(batch))]

    default_collated = default_collate(to_default_collate)

    return default_collated, event_idxs, events



class PointCloudDataModule(L.LightningDataModule):

    def __init__(
        self,
        input_dir,
        input_features,
        position_features,
        num_events: list[int]=[1,1,1],
        use_csv: bool=False,
        event_prefix="",
        variable_with_prefix: bool = False,
        node_scales: list[float] = [],
        points_per_batch: int = 1000,
        knn: list[int] = [],
        sampling_ratio: list[int] = [],
        weighting: list[dict] = [{}],
        subsampling="random",
        batch_size: int=2,
        num_workers: int=16,
    ):
        super().__init__()

        self.input_dir = input_dir
        self.input_features = input_features
        self.position_features = position_features
        self.num_events = num_events
        self.use_csv = use_csv
        self.event_prefix = event_prefix
        self.variable_with_prefix = variable_with_prefix
        self.node_scales = node_scales
        self.points_per_batch = points_per_batch
        self.knn = knn
        self.sampling_ratio = sampling_ratio
        self.weighting = weighting
        self.subsampling = subsampling
        self.batch_size=batch_size
        self.num_workers=num_workers
    
    def setup(self, stage: str) -> None:
        
        for dataset_name, num_events in zip(['trainset', 'valset', 'testset'], self.num_events):
            dataset = PointCloudDataset(
                self.input_dir,
                self.input_features,
                self.position_features,
                dataset_name,
                num_events,
                self.use_csv,
                self.event_prefix,
                self.variable_with_prefix,
                self.node_scales,
                self.points_per_batch,
                self.knn,
                self.sampling_ratio,
                self.weighting,
                stage,
                subsampling=self.subsampling
            )

            setattr(self, dataset_name, dataset)
        
    def _dataloader(self, dataset, shuffle=False, collate_fn=None) -> DataLoader:
        if collate_fn is not None:
        
            return DataLoader(
                getattr(self, dataset),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                shuffle=shuffle,
                collate_fn=collate_fn
            )
        
        else:

            return DataLoader(
                getattr(self, dataset),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True,
                shuffle=shuffle
            )


    def train_dataloader(self):
        return self._dataloader('trainset', True)
    
    def val_dataloader(self):
        return self._dataloader("valset")

    def test_dataloader(self):
        return self._dataloader("testset")
    
    def predict_dataloader(self):
        return [
            self._dataloader("trainset", collate_fn=predict_collate_fn),
            self._dataloader("valset", collate_fn=predict_collate_fn),
            self._dataloader("testset", collate_fn=predict_collate_fn)
        ]


        