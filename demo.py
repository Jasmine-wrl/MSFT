from argparse import ArgumentParser

# import utils
import torch
from models.basic_model import CDEvaluator

import os

################################### utils 内容######### 
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import utils

import data_config
from datasets.CD_dataset import CDDataset

#####获取训练集
def get_loader(data_name, img_size=256, batch_size=8, split='test',
               is_train=False, dataset='CDDataset'):
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform

    if dataset == 'CDDataset':
        data_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=img_size, is_train=is_train,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset])'
            % dataset)

    shuffle = is_train
    dataloader = DataLoader(data_set, batch_size=batch_size,
                                 shuffle=shuffle, num_workers=4)

    return dataloader

######获取训练集和验证集
def get_loaders(args):

    data_name = args.data_name
    dataConfig = data_config.DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir
    label_transform = dataConfig.label_transform
    split = args.split
    split_val = 'val'
    if hasattr(args, 'split_val'):
        split_val = args.split_val
    if args.dataset == 'CDDataset':
        training_set = CDDataset(root_dir=root_dir, split=split,
                                 img_size=args.img_size,is_train=True,
                                 label_transform=label_transform)
        val_set = CDDataset(root_dir=root_dir, split=split_val,
                                 img_size=args.img_size,is_train=False,
                                 label_transform=label_transform)
    else:
        raise NotImplementedError(
            'Wrong dataset name %s (choose one from [CDDataset,])'
            % args.dataset)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)
                   for x in ['train', 'val']}

    return dataloaders


def make_numpy_grid(tensor_data, pad_value=0,padding=0):
    tensor_data = tensor_data.detach()#####x.data和x.detach()新分离出来的tensor的requires_grad=False，
    ##################################即不可求导时两者之间没有区别，但是当requires_grad=True的时候的两者之间的是有不同：
    ################################## x.data不能被autograd追踪求微分，但是x.detach可以被autograd()追踪求导。
    vis = torchvision.utils.make_grid(tensor_data, pad_value=pad_value,padding=padding)
    vis = np.array(vis.cpu()).transpose((1,2,0))
    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)
    return vis


def de_norm(tensor_data):
    return tensor_data * 0.5 + 0.5


def get_device(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

################################################

"""
quick start

sample files in ./samples
save prediction files in the ./samples/predict

"""


def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='BIT_LEVIR', type=str)
    # parser.add_argument('--project_name', default='CD_base_transformer_pos_s4_dd8_dedim8_LEVIR_b4_lr0.01_train_val_100_linear', type=str)  ##与训练对应
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    # parser.add_argument('--output_folder', default='samples/predict', type=str)
    parser.add_argument('--output_folder', default='LEVIR-CD/predict', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    # parser.add_argument('--data_name', default='quick_start', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)


    parser.add_argument('--batch_size', default=1, type=int)
    # parser.add_argument('--split', default="demo", type=str)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--img_size', default=1024, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4fpn_diff_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = get_args()
    # utils.get_device(args)
    get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids)>0
                        else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')

    # data_loader = utils.get_loader(args.data_name, img_size=args.img_size,
                                #    batch_size=args.batch_size,
                                #    split=args.split, is_train=False)

    data_loader = get_loader(args.data_name, img_size=args.img_size,
                                    batch_size=args.batch_size,
                                    split=args.split, is_train=False)
    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    model.eval()

    for i, batch in enumerate(data_loader):
        name = batch['name']
        print('process: %s' % name)
        score_map = model._forward_pass(batch)
        model._save_predictions()




