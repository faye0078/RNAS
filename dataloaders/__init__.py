from dataloaders.datasets import GID
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
    data_path = get_data_path(args.dataset)
    num_class = 5

    composed_trn = transforms.Compose(
        [
            RandomMirror(),
            RandomCrop(args.crop_size),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    composed_val = transforms.Compose(
        [

            RandomCrop(args.crop_size),
            Normalise(*args.normalise_params),
            ToTensor(),
        ]
    )
    composed_test = transforms.Compose(
        [
            RandomCrop(args.crop_size),
            Normalise(*args.normalise_params),
            ToTensor(),
        ])

    train_set = GID.GIDDataset(stage="train",
                                data_file=data_path['mini_train_list'],
                                data_dir=data_path['dir'],
                                transform_trn=composed_trn,)
    val_set = GID.GIDDataset(stage="val",
                                data_file=data_path['val_list'],
                                data_dir=data_path['dir'],
                                transform_val=composed_val,)
    test_set = GID.GIDDataset(stage="test",
                                data_file=data_path['test_list'],
                                data_dir=data_path['dir'],
                                transform_test=composed_test,)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader, num_class
