######

from argparse import ArgumentParser
import utils
import torch
from models.basic_model import CDEvaluator

import os

def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    # parser.add_argument('--project_name', default='BIT_LEVIR', type=str)
    parser.add_argument('--project_name', default='CD_base_transformer_pos_s4_dd8_dedim8_LEVIR_b4_lr0.01_train_val_100_linear', type=str)  ##与训练对应
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
    # parser.add_argument('--output_folder', default='samples/predict', type=str)
    parser.add_argument('--output_folder', default='LEVIR-CD/predict', type=str)

    # # data
    # parser.add_argument('--num_workers', default=0, type=int)
    # parser.add_argument('--dataset', default='CDDataset', type=str)
    # # parser.add_argument('--data_name', default='quick_start', type=str)
    # parser.add_argument('--data_name', default='LEVIR', type=str)


    # parser.add_argument('--batch_size', default=1, type=int)
    # # parser.add_argument('--split', default="demo", type=str)
    # parser.add_argument('--split', default="test", type=str)
    # parser.add_argument('--img_size', default=1024, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8_dedim8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    utils.get_device(args)
    
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids)>0
                        else "cpu")

    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)      
    assert os.path.exists(args.checkpoint_dir), "file {} does not exist.".format(args.checkpoint_dir)

    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)

    # # 2. 打印网络及按名称查看子网络
    # for name, module in model.named_modules():
    #     print(name, module)

    ######本模型
    print(model.net_G)


if __name__ == '__main__':
    main()



