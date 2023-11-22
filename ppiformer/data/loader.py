from typing import Optional

from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader

from ppiformer.data.transforms import *
from ppiformer.data.dataset import PPIInMemoryDataset


# TODO class PPIDataLoader(DataLoader)
# - put `get_dataloader` logic into constructor
# - extend `Collater` to properly collate `node_id` of Graphein strings into a single list


def get_dataloader(
    dataset: str,
    pretransform: Iterable[BaseTransform] = tuple(),
    prefilter: Iterable[BaseTransform] = tuple(),
    transform: Iterable[BaseTransform] = tuple(),
    dataset_max_workers: Optional[int] = None,
    shuffle: bool = True,
    batch_size: int = 8,
    num_workers: int = 0,
    fresh: bool = False,
    deterministic: bool = False,
    verbose: bool = True,
    **kwargs
) -> DataLoader:
    split, fold = parse_dataset_argument(dataset)

    # Make all transforms determenisitic if debugging
    if deterministic:
        for t in pretransform + transform:
            if isinstance(t, StochasticTransform):
                t.deterministic = True

    # Load dataset
    dataset_name = dataset
    dataset = PPIInMemoryDataset(
        split=split,
        fold=fold,
        pre_transform=T.Compose(pretransform),
        pre_filter=ComposeFilters(prefilter),
        transform=T.Compose(transform),
        fresh=fresh,
        max_workers=dataset_max_workers
    )
    if verbose:
        print(f'{dataset_name} loaded: {dataset}')

    # Get data attributes produced by pre-processing that have their own batching (different
    # from node batching)
    follow_batch = []
    for t in pretransform + transform:
        if hasattr(t, 'follow_batch_attrs'):
            follow_batch.extend(t.follow_batch_attrs())

    # Convert to dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=shuffle,
        follow_batch=follow_batch,
        **kwargs
    )
    return dataloader


def parse_dataset_argument(arg: str) -> tuple[str, str]:
    split, fold = arg.strip().replace(' ', '').split(',')
    return split, fold
