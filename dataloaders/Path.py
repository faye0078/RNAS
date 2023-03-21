from collections import OrderedDict
def get_data_path(dataset):
    if dataset == 'GID':
        Path = OrderedDict()
        Path['dir'] = "../data/512"
        Path['train_list'] = "./data/lists/GID/rs_train.lst"
        Path['val_list'] = "./data/lists/GID/rs_val.lst"
        Path['test_list'] = "./data/lists/GID/rs_test.lst"
        Path['mini_train_list'] = "./data/lists/GID/mini_rs_train.lst"
    if dataset == 'FBP':
        Path = OrderedDict()
        Path['dir'] = "../../data"
        Path['train_list'] = "./data/lists/FBP/fbp_ori_train.lst"
        Path['val_list'] = "./data/lists/FBP/fbp_ori_val.lst"
        Path['test_list'] = "./data/lists/FBP/fbp_ori_test.lst"
    return Path