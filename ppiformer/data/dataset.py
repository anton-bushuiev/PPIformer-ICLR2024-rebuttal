import os

from tqdm import tqdm
import torch
import torch.multiprocessing
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, separate, collate


from ppiref.split import read_fold
from ppiformer.data.transforms import *
from ppiformer.definitions import PPIFORMER_PYG_DATA_CACHE_DIR
from  concurrent.futures import ProcessPoolExecutor
import warnings
import graphein

class PPIInMemoryDataset(InMemoryDataset):

    n_classes = 20  # Number of amino acid types

    def __init__(  # TODO Set default params
        self,
        split: str,  # PPIRef split
        fold: str,  # PPIRef fold
        verbose: bool = True,
        fresh: bool = False,
        pre_transform=T.Compose([
            PDBToPyGPretransform()
            # DDGLabelPretransform(),
            # CleanPretransform()
        ]),
        pre_filter=ComposeFilters([
            # PPISizeFilter(max_nodes=200),
            # DDGFilter()
        ]),
        transform=T.Compose([
            # DeepCopyTransform(),
            # MaskedModelingTransform()
            # CompleteGraphTransform(),
        ]),
        max_workers: Optional[int] = None,
        skip_data_on_processing_errors: bool = False
    ):
        # Process input args
        self.split = split
        self.fold = fold
        self.verbose = verbose
        self.max_workers = max_workers if max_workers is not None else os.cpu_count() - 2
        self.skip_data_on_processing_errors = skip_data_on_processing_errors
        
        self.root = PPIFORMER_PYG_DATA_CACHE_DIR
        self.raw_data_file_names = read_fold(self.split, self.fold)

        # Clean cache
        self.clean_cache(transforms_only=True)
        # if fresh:
        #     self.clean_cache()
        # else:
        #     self.clean_cache(transforms_only=True)

        # Init
        super().__init__(self.root, transform, pre_transform, pre_filter, verbose)

        # Load data
        data_list = []
        if len(self.processed_paths) > 1:
            for p in self.processed_paths:
                data_collated, slices = torch.load(p)
                for idx in range(slices['x'].shape[0] - 1):  # estimate n. graphs in dataset by coordinate slices
                    data = separate.separate(cls=data_collated.__class__,batch=data_collated, slice_dict=slices, idx=idx, decrement=False)
                    data_list.append(data)
            print(f'Loaded {len(data_list)} graphs from {len(self.processed_paths)} files')
            print('Collating...')
            self.data, self.slices = self.collate(data_list=data_list)
        else:
            if fresh:
                self.process()
            else:
                self.data, self.slices = torch.load(self.processed_paths[0])
            
        
        # Init attrbiutes
        self.n_features = self._data.f.shape[1]
        self.n_coords = self._data.x.shape[1]

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return self.raw_data_file_names

    @property
    def processed_file_names(self):
        if os.path.isfile(self.root / f'ppi_inmemory_dataset_{self.split}_{self.fold}.pt'):
            return [f'ppi_inmemory_dataset_{self.split}_{self.fold}.pt']
        elif '+' in self.fold:
            #parse the fold name
            fold_names = self.fold.split('+')
            files = [f'ppi_inmemory_dataset_{self.split}_{fold_name}.pt' for fold_name in fold_names]
            if all([os.path.isfile(self.root / f) for f in files]):
                return files
            else:
                return [f'ppi_inmemory_dataset_{self.split}_{self.fold}.pt']
        else:
            return [f'ppi_inmemory_dataset_{self.split}_{self.fold}.pt']

    def pre_transform_chunk(self, chunk):
        data_list = []
        graphein.verbose(enabled=False)
        warnings.simplefilter(action='ignore', category=FutureWarning)
        # for path in tqdm(chunk, file=sys.stderr, desc=f'Process {os.getpid()} working'):
        for path in tqdm(chunk, desc=f'Process {os.getpid()} preparing data'):
            try:
                data = self.pre_transform(path)
            except Exception as e:
                if not self.skip_data_on_processing_errors:
                    raise e
                else:
                    print(f'Process {os.getpid()} failed on {path}\n{e}')  # TODO Trace
            else:
                data_list.append(data)
        return data_list

    def process(self):
        # Read and process data points into the list of `Data`.
        # NOTE: the order of pre-filter and pre-transform is opposite to PyG docs
        if self.pre_transform is not None:
            torch.multiprocessing.set_sharing_strategy('file_system')
            n_chunks = (self.max_workers - 2)
            chunksize = max(1, len(self.raw_paths) // n_chunks)
            chunks = [self.raw_paths[i:i + chunksize] for i in range(0, len(self.raw_paths), chunksize)]

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                data_list = list(executor.map(self.pre_transform_chunk, chunks))
            data_list = sum(data_list, start=[])
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if len(self.raw_data_file_names) != len(data_list):
            warnings.warn(f'Only {len(data_list)} our of {len(self.raw_data_file_names)} files were read and processed.')

        # Ensure safe collate (at least partially) by dropping data points with inconsistent
        # attribute types
        bad_idx = []
        for attr in data_list[0].stores[0].keys():
            expected_type = type(getattr(data_list[0], attr))
            for i, data in enumerate(data_list):
                real_type = type(getattr(data, attr))
                if real_type != expected_type:
                    bad_idx.append(i)
                    msg = f'Inconsistent attribute type for {attr} in {data.path}. Real: {real_type}, expected: {expected_type}.'
                    if self.skip_data_on_processing_errors:
                        print(msg)
                    else:
                        raise ValueError(msg)
        bad_idx = set(bad_idx)
        data_list = [data for i, data in enumerate(data_list) if i not in bad_idx]

        # Collate and save
        data, slices = self.collate(data_list)
        self.data, self.slices = data, slices
        torch.save((data, slices), self.processed_paths[0])

    def clean_cache(self, transforms_only: bool = False) -> None:
        processed_dir = self.root / 'processed'
        cache_files = [
            processed_dir / 'pre_transform.pt',
            processed_dir / 'pre_filter.pt'
        ]
        if not transforms_only:
            cache_files += [processed_dir / name for name in self.processed_file_names]

        for path in cache_files:
            path.unlink(missing_ok=True)
        
    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ''
        retval = f'{self.__class__.__name__}({arg_repr})'
        if hasattr(self._data, 'n_muts'):
            n_muts = self._data.n_muts
            n_muts = n_muts.sum().item() if isinstance(n_muts, torch.Tensor) else n_muts
            retval = retval[:-1] + f', n_muts={n_muts})'
        return retval
