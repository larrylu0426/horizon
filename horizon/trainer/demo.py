import torch
import torchvision
from tqdm import tqdm
import wandb

from horizon.base.trainer import BaseTrainer
import horizon.data_loader as module_data
from horizon.utils.visualizer import get_masked_image


class Demo(BaseTrainer):

    def __init__(self, model, criterion, metric_ftns, optimizer, lr_scheduler,
                 config, device):
        super().__init__(model, criterion, metric_ftns, optimizer,
                         lr_scheduler, config, device)

    def _add_data_loaders(self):
        train_data_loader = self.config.init_obj('dataset',
                                                 module_data,
                                                 train=True,
                                                 validation_split=0.1)
        valid_data_loader = train_data_loader.split_validation()
        test_data_loader = self.config.init_obj('dataset',
                                                module_data,
                                                train=False)
        return train_data_loader, valid_data_loader, test_data_loader

    def _train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        for _, (data, target) in tqdm(enumerate(self.train_data),
                                      total=len(self.train_data),
                                      leave=True):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            loss.backward()
            self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return self.train_metrics.result()

    def _valid_epoch(self):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                if batch_idx == 0:
                    pred = torch.argmax(output, dim=1)
                    self._log_image_table(data, pred, target)
                loss = self.criterion(output, target)

                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__,
                                              met(output, target))

        return self.valid_metrics.result()

    def _test_model(self):
        model = self.best_model
        model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for _, (data, target) in enumerate(tqdm(self.test_data)):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))

        return self.test_metrics.result()

    def _log_image_table(self, data, predicted, labels):
        label_map = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
        table = wandb.Table(columns=["image",  "label", "prediction"])
        for img, pred, targ in zip(
                data.to("cpu"), predicted.to("cpu"), labels.to("cpu")):
            table.add_data(
                wandb.Image(img),
                label_map[targ], label_map[pred])
        wandb.log({"predictions_table": table}, commit=False)