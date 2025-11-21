"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import copy


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
            dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        dataset_class = find_dataset_using_name(opt.dataset_mode)

        if opt.phase == 'test':
            # For test phase, create single dataset
            self.test_dataset = dataset_class(opt)
            print("test dataset [%s] was created" %
                  type(self.test_dataset).__name__)

            # Test dataloader with specific settings
            self.test_dataloader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=int(opt.num_threads))
        else:
            # For train/val phases, create both datasets
            # Create training dataset
            opt_train = copy.deepcopy(opt)
            opt_train.phase = 'train'
            self.train_dataset = dataset_class(opt_train)
            print("training dataset [%s] was created" %
                  type(self.train_dataset).__name__)

            # Create validation dataset
            opt_val = copy.deepcopy(opt)
            opt_val.phase = 'val'
            opt_val.isTrain = False
            opt_val.no_flip = True
            self.val_dataset = dataset_class(opt_val)
            print("validation dataset [%s] was created" %
                  type(self.val_dataset).__name__)

            # Create dataloaders
            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.num_threads))

            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=opt.batch_size,
                shuffle=False,  # No need to shuffle validation data
                num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        if self.opt.phase == 'test':
            return len(self.test_dataset)
        elif self.opt.phase == 'val':
            return len(self.val_dataset)
        return min(len(self.train_dataset), self.opt.max_dataset_size)
