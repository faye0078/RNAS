from dataloaders.datasets import GID
from dataloaders.datasets import FBP
from dataloaders.datasets import normal_FBP
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append("../")
from dataloaders.Path import get_data_path

from dataloaders.datasets.GID import (
    Normalise,
    RandomCrop,
    RandomMirror,
    ToTensor,
)
from torchvision import transforms
def make_data_loader(args, **kwargs):
    if args.dataset == "GID":
        Dataset = GID.GIDDataset
        num_class = 5
    if args.dataset == "FBP":
        Dataset = FBP.FBPDataset
        num_class = 24
    if args.dataset == "normal_FBP":
        Dataset = normal_FBP.FBPDataset
        num_class = 24
    data_path = get_data_path(args.dataset)
    composed_trn = transforms.Compose(
        [
            RandomMirror(),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    composed_val = transforms.Compose(
        [
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    composed_test = transforms.Compose(
        [
            Normalise(*args.normalise_params),
            ToTensor(),
        ])
    train_set = Dataset(stage='train',
                        # image_size_rand=[256,900],
                        data_file=data_path['train_list'],
                        data_dir=data_path['dir'],
                        transform_trn=composed_trn,)
    val_set = Dataset(stage='val',
                        # image_size_rand=[256,2048],
                        data_file=data_path['val_list'],
                        data_dir=data_path['dir'],
                        transform_val=composed_val,)
    test_set = Dataset(stage='test',
                        # image_size_rand=[256,900],
                        data_file=data_path['test_list'],
                        data_dir=data_path['dir'],
                        transform_test=composed_test,)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, num_class

def make_search_data_loader(args, **kwargs):
    if args.dataset == "GID":
        Dataset = GID.GIDDataset
        num_class = 5
    if args.dataset == "FBP":
        Dataset = FBP.FBPDataset
        num_class = 24
    data_path = get_data_path(args.dataset)
    composed_trn = transforms.Compose(
        [
            RandomMirror(),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    composed_val = transforms.Compose(
        [
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    proxy_train_set = Dataset(stage='train',
                        image_size=kwargs['image_size'],
                        image_size_rand=None,
                        data_file=data_path['train_mini_list'],
                        data_dir=data_path['dir'],
                        transform_trn=composed_trn,)
    val_set = Dataset(stage='val',
                        image_size=kwargs['image_size'],
                        image_size_rand=None,
                        data_file=data_path['val_mini_list'],
                        data_dir=data_path['dir'],
                        transform_val=composed_val,)
    proxy_train_loader = DataLoader(proxy_train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    return proxy_train_loader, val_loader, num_class
