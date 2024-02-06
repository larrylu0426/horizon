import torch
from tqdm import tqdm
import wandb

from horizon.base.trainer import BaseTrainer
import horizon.data_loader as module_data


class Demo(BaseTrainer):

    def __init__(self, config, model, device):
        super().__init__(config, model, device)

    def _add_data_loaders(self):
        train_data_loader = self.config.init_obj('dataset', module_data)
        val_data_loader = train_data_loader.split_validation()
        test_data_loader = train_data_loader.split_test()
        return train_data_loader, val_data_loader, test_data_loader

    def _train(self):
        self.model.train()
        self.train_metrics.reset()
        for _, (data, target) in tqdm(enumerate(self.train_data),
                                      total=len(self.train_data),
                                      leave=True):
            self.optimizer.zero_grad()
            data, target = data.to(self.device), target.to(self.device)
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

    def _valid(self, model):
        model.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for _, (data, target) in enumerate(self.val_data):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                self.val_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.val_metrics.update(met.__name__, met(output, target))

        return self.val_metrics.result()

    def _test(self, model):
        model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.test_data)):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(output, target))
                if batch_idx == 0 and self.config.wandb:
                    pred = torch.argmax(output, dim=1)
                    self._log_image_table(data, pred, target)

        return self.test_metrics.result()

    def _log_image_table(self, data, predicted, labels):
        label_map = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
            'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
            'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
            'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
            'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
            'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
            'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
            'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
            'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
            'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
            'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
            'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
            'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
            'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
            'worm'
        ]
        table = wandb.Table(columns=["image", "label", "prediction"])
        for img, pred, targ in zip(data.to("cpu"), predicted.to("cpu"),
                                   labels.to("cpu")):
            table.add_data(wandb.Image(img), label_map[targ], label_map[pred])
        wandb.log({"predictions_table": table}, commit=False)
