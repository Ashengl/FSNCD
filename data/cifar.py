from torchvision.datasets import CIFAR10, CIFAR100
from copy import deepcopy
import numpy as np

from data.data_utils import subsample_instances
from config import cifar_10_root, cifar_100_root


class CustomCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):

        super(CustomCIFAR10, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


class CustomCIFAR100(CIFAR100):

    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx

    def __len__(self):
        return len(self.targets)


def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None

def merge_dataset(dataset1, dataset2):

    dataset1.data = np.concatenate((dataset1.data, dataset2.data), axis=0)
    dataset1.targets.extend(dataset2.targets)
    dataset1.uq_idxs = np.concatenate((dataset1.uq_idxs, dataset2.uq_idxs), axis=0)

    return dataset1

def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_cifar_10_datasets(train_transform, test_transform, train_classes=(0, 1, 8, 9),
                       prop_train_labels=0.8, split_train_val=False, seed=0, args=None):

    np.random.seed(seed)

    training_set = CustomCIFAR10(root=cifar_10_root, transform=train_transform, train=True)
    test_set = CustomCIFAR10(root=cifar_10_root, transform=test_transform, train=False)

    train_dataset_1 = subsample_classes(deepcopy(training_set), include_classes=args.train_classes)
    test_dataset_1 = subsample_classes(deepcopy(training_set), include_classes=args.unlabeled_classes)

    train_dataset_2 = subsample_classes(deepcopy(test_set), include_classes=args.train_classes)
    test_dataset_2 = subsample_classes(deepcopy(test_set), include_classes=args.unlabeled_classes)

    train_dataset = merge_dataset(train_dataset_1, train_dataset_2)
    test_dataset = merge_dataset(test_dataset_1, test_dataset_2)

    train_idxs, val_idxs = get_train_val_indices(train_dataset)
    train_dataset_split = subsample_dataset(deepcopy(train_dataset), train_idxs)
    val_dataset_split = subsample_dataset(deepcopy(train_dataset), val_idxs)
    val_dataset_split.transform = test_transform

    train_dataset_labelled = train_dataset_split if split_train_val else train_dataset
    val_dataset_labelled = val_dataset_split if split_train_val else None
    train_dataset_labelled.label = train_dataset_labelled.targets
    if split_train_val:
        val_dataset_labelled.label = val_dataset_labelled.targets
    test_dataset.label = test_dataset.targets


    all_datasets = {
        'train': train_dataset_labelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_cifar_100_datasets(train_transform, test_transform, train_classes=range(80),
                       prop_train_labels=0.8, split_train_val=False, seed=0, args=None):

    np.random.seed(seed)

    training_set = CustomCIFAR100(root=cifar_100_root, transform=train_transform, train=True)
    test_set = CustomCIFAR100(root=cifar_100_root, transform=test_transform, train=False)

    train_dataset_1 = subsample_classes(deepcopy(training_set), include_classes=args.train_classes)
    test_dataset_1 = subsample_classes(deepcopy(training_set), include_classes=args.unlabeled_classes)

    train_dataset_2 = subsample_classes(deepcopy(test_set), include_classes=args.train_classes)
    test_dataset_2 = subsample_classes(deepcopy(test_set), include_classes=args.unlabeled_classes)

    train_dataset = merge_dataset(train_dataset_1, train_dataset_2)
    test_dataset = merge_dataset(test_dataset_1, test_dataset_2)

    train_idxs, val_idxs = get_train_val_indices(train_dataset)
    train_dataset_split = subsample_dataset(deepcopy(train_dataset), train_idxs)
    val_dataset_split = subsample_dataset(deepcopy(train_dataset), val_idxs)
    val_dataset_split.transform = test_transform

    train_dataset_labelled = train_dataset_split if split_train_val else train_dataset
    val_dataset_labelled = val_dataset_split if split_train_val else None
    train_dataset_labelled.label = train_dataset_labelled.targets
    if split_train_val:
        val_dataset_labelled.label = val_dataset_labelled.targets
    test_dataset.label = test_dataset.targets


    all_datasets = {
        'train': train_dataset_labelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_cifar_100_datasets(None, None, split_train_val=False,
                         train_classes=range(80), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')