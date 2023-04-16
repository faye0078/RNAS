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
        Path['train_mini_list'] = "./data/lists/FBP/fbp_ori_train_mini.lst"
        Path['val_list'] = "./data/lists/FBP/fbp_ori_val.lst"
        Path['val_mini_list'] = "./data/lists/FBP/fbp_ori_val_mini.lst"
        Path['test_list'] = "./data/lists/FBP/fbp_ori_test.lst"
        
    if dataset == 'normal_FBP':
        Path = OrderedDict()
        Path['dir'] = "/mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/wy/data"
        Path['train_list'] = "./data/lists/normal_FBP/fbp_756_train.lst"
        Path['val_list'] = "./data/lists/normal_FBP/fbp_756_val.lst"
        Path['test_list'] = "./data/lists/normal_FBP/fbp_756_test.lst"
        
    return Path