# Horizon: A PyTorch-based deep learning framework with WandB

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

This framework is developed based on [Pytorch-template](https://github.com/victoresque/pytorch-template).

## Features
* Clear folder structure which is suitable for many deep learning projects.
* Customizable command line options for more convenient parameter tuning.
* `.yaml` config file (followed [wandb.config](https://docs.wandb.ai/guides/track/config)) support for convenient parameter tuning.
* Using [wandb.Artifact](https://docs.wandb.ai/ref/python/artifact) to save trained models.
* Abstract base classes for faster development:
  * `BaseTrainer` combines with WandB to handle the training lifecycle, logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  horizon/
  │
  ├── run.py - main script to start train/val/test
  ├── generator.py - initialize new project with template files
  │
  ├── data/ - default directory for storing input data
  |
  ├── saved/ - default directory for storing experiments
  │
  ├── config/ - default directory for storing configuration files
  │   └── defaults.yaml - holds configuration for training
  │
  ├── horizon/ - root of package
  │   ├── base/ - abstract base classes
  │     ├── base_data_loader.py
  │     ├── base_model.py
  │     └── base_trainer.py
  │   ├── data_loader/ - anything about data loading goes here
  │   ├── model/ - trained models are saved here
  |   ├── trainer/ - trainers are saved here
  |   └── utils/ - small utility functions
  |     ├── config.py - class to handle config file
  |     ├── const.py
  │     ├── logger.py  
  │     ├── loss.py
  │     ├── metric.py
  │     ├── util.py
  |     └── ...
  ```

## Usage
First, clone this repository
```bash
git clone https://github.com/larrylu0426/horizon.git
```
Then, install the dependencies
```bash
pip install -r requirements.txt
```
The code in this repo is an CIFAR100-ViT example of the frameworks. Try to test without WandB's features.
```
python run.py -m train
```

### Config file format
Config files are in `.yaml` format:
```YAML
arch:
  value: 
    name: VIT # name of model architecture to train

dataset:
  value:
    name: CIFAR100 # selecting data loader
    args:
      root: data/CIFAR100 # dataset path
      batch_size: 256 # batch size
      shuffle: false # shuffle training data before splitting
      num_workers: 16 # number of cpu processes to be used for data loading
      val_split: 0.1
      test_split: 0.2

loss:
  value: cross_entropy # loss function

optimizer:
  value:
    name: SGD # selecting optimizer
    args:
      lr: 0.1 # learning rate
      weight_decay: 1.0e-4 # (optional) weight decay
      momentum: 0.9
lr_scheduler:
  value:
    name: StepLR # learning rate scheduler
    args:
      step_size: 20
      gamma: 0.1
metrics:
  value:
    - accuracy # list of metrics to evaluate
trainer:
  value:
    name: Demo
    seed: 0 # the value of seed
    use_gpu: true # use GPU or CPU
    n_gpu: 0 # ids of GPUs to use
    epochs: 60 # number of training epochs
    monitor: max val_accuracy # mode and metric for model performance monitoring. set 'off' to disable.
    early_stop: 10 # number of epochs to wait before early stop. set 0 to disable.
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `xxx.yaml` config files, then run:

  ```
  python run.py -c config/xxx.yaml
  ```

### Switch train or test mode
To train a model, run:

  ```
  python run.py -m train
  ```
To train a model from a checkpoint, run:

  ```
  python run.py -m train -r <checkpoint_path>
  ```
To test a model, run:

  ```
  python run.py -m test -r <checkpoint_path>
  ```


### Using WandB
run:

  ```
  python run.py -w True
  ```
The `project` name displayed in your WandB page is your project folder name, and the `run` name is formatted as '{dataloader_name}\_{model_name}\_{MMDD}\_{timestamp}'. If you want to use a self-defined project name, please run:
  ```
  python run.py -w True -n XXX
  ```
## Customization

### Project initialization
Use the `generator.py` script to make your new project directory.
`python generator.py path/project_name` then a new project folder named 'project_name' will be made.
This script will filter out unneccessary files like cache, git files or readme file. 


### Data Loader
* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Generating validation/test data loader by calling
    `BaseDataLoader.split_validation()` and `BaseDataLoader.split_test()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:
  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```
* **Example**

  Please refer to `data_loader/cifar100.py` for a CIFAR100 data loading example.

### Model
* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/vit.py` for a ViT example.

### Loss
Custom loss functions can be implemented in 'utils/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.
```yaml
loss:
  value: cross_entropy
```
or
```yaml
loss:
  value: 
    - cross_entropy
    - nll_loss
```
For the scene which may use multiple loss function, you can use the specific loss function by: `loss['name']`
### Metrics
Metric functions are located in 'utils/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
```yaml
metrics:
  value:
    - accuracy
```
### Trainer
* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Lifecycle logging
    * Checkpoint saving with WandB
    * Checkpoint resuming
    * Reconfigurable performance monitoring for choose current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will record the result of the best model in the `wandb.summary` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train()` for your training process, if you need validation and test then you can implement `_valid()` and `_test()` with provided `valid_dataloader` and `test_dataloader`.

* **Example**

  Please refer to `trainer/demo.py` for CIFAR100-ViT training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.


## License
This project is licensed under the MIT License. See  LICENSE for more details
