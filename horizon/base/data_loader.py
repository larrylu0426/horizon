from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataloader import default_collate


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle,
                 val_split,
                 test_split,
                 num_workers,
                 collate_fn=default_collate):
        self.dataset = dataset
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.n_datasets = len(dataset)

        self.train_data, self.val_data, self.test_data = self._split_dataset(
            self.val_split, self.test_split)

        self.init_kwargs = {
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': True,
        }

        super().__init__(dataset=self.train_data,
                         shuffle=shuffle,
                         **self.init_kwargs)

    def _split_dataset(self, val_split, test_split):
        if val_split == 0.0 and test_split == 0.0:
            # only give train sampler in the super class constructor
            return (self.dataset, None, None)

        len_val = int(self.n_datasets * val_split)
        len_test = int(self.n_datasets * test_split)
        len_train = self.n_datasets - len_val - len_test
        train_data, val_data, test_data = random_split(
            self.dataset, [len_train, len_val, len_test])

        return train_data, val_data, test_data

    def split_validation(self):
        if self.val_data is None:
            return None
        else:
            return DataLoader(dataset=self.val_data,
                              shuffle=False,
                              **self.init_kwargs)

    def split_test(self):
        if self.test_data is None:
            return None
        else:
            return DataLoader(dataset=self.test_data,
                              shuffle=False,
                              **self.init_kwargs)
