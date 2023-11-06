from torchvision import datasets, transforms
from horizon.base.data_loader import BaseDataLoader


class CIFAR100(BaseDataLoader):
    """
    CIFAR100 data loading demo using BaseDataLoader
    """

    def __init__(self,
                 root,
                 batch_size,
                 shuffle=True,
                 val_split=0.0,
                 test_split=0.0,
                 num_workers=1):
        trsfm = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))])
        self.root = root
        self.dataset = datasets.CIFAR100(self.root,
                                         download=True,
                                         transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, val_split,
                         test_split, num_workers)
