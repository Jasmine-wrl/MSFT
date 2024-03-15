import numpy as np
import cv2
import torch
import os
import utils

from argparse import ArgumentParser
from models.basic_model_vis import CDEvaluator

import torchvision.transforms as transforms

import pytorch_grad_cam 
from pytorch_grad_cam.utils.image import show_cam_on_image



######
def get_args():
    parser = ArgumentParser()

    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    ############################需要修改对应
    parser.add_argument('--project_name', default='CD_base_transformer_pos_s4fpn_diff_dd8_e6d2_sk_LEVIR_b16_lr0.0001_adam_train_val_100_linear_nw4', type=str) 
    parser.add_argument('--output_folder', default='./DataVis/', type=str)
    parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    # model
    parser.add_argument('--n_class', default=2, type=int)

    ############################需要修改对应
    parser.add_argument('--net_G', default='base_transformer_pos_s4fpn_diff_dd8_e6d2_sk', type=str ,help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    return args


def get_model():
    args = get_args()
    utils.get_device(args)

    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)      
    
    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)

    # # 2. 打印网络及按名称查看子网络
    # for name, module in model.named_modules():
    #     print(name, module)

    ######本模型
    return model.net_G

    
def vis(path_out):
    # 1.定义模型结构，选取要可视化的层
    mymodel = get_model()
    # traget_layers = [resnet18.layer4[1].bn2]
 
    traget_layer1 = [mymodel.fpn]
    traget_layer2 = [mymodel.transformer] 
    traget_layer3 = [mymodel.transformer_decoder] 
    traget_layer4 = [mymodel.sk]
    traget_layer5 = [mymodel.diff] 

    # 2.读取图片，将图片转为RGB

    origin_img1 = cv2.imread('/home/wangruilan/CD/Datasets/LEVIR-CD-256-v1/A/test_7_16.png')
    origin_img2 = cv2.imread('/home/wangruilan/CD/Datasets/LEVIR-CD-256-v1/B/test_7_16.png')
    rgb_img1 = cv2.cvtColor(origin_img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(origin_img2, cv2.COLOR_BGR2RGB)


    # 3.图片预处理：resize ToTensor()
    trans = transforms.ToTensor()
    crop_img1 = trans(rgb_img1)
    crop_img2 = trans(rgb_img2)


    net_input1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(crop_img1).unsqueeze(0)
    net_input2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(crop_img2).unsqueeze(0)

    # 4.将裁剪后的Tensor格式的图像转为numpy格式，便于可视化
    canvas_img1 = (crop_img1*255).byte().numpy().transpose(1, 2, 0)
    canvas_img1 = cv2.cvtColor(canvas_img1, cv2.COLOR_RGB2BGR)

    canvas_img2 = (crop_img2*255).byte().numpy().transpose(1, 2, 0)
    canvas_img2 = cv2.cvtColor(canvas_img2, cv2.COLOR_RGB2BGR)


    # 5.实例化cam
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=mymodel, target_layers=traget_layer3)
    # grayscale_cam1 = cam(net_input1)
    # grayscale_cam1 = grayscale_cam1[0, :]
    # grayscale_cam2 = cam(net_input2)
    # grayscale_cam2 = grayscale_cam2[0, :]


    grayscale_cam1, grayscale_cam2  = cam(net_input1, net_input2)
    grayscale_cam1 = grayscale_cam1[0, :]
    grayscale_cam2 = grayscale_cam2[0, :]


    # 6.将feature map与原图叠加并可视化
    src_img1 = np.float32(canvas_img1) / 255
    src_img2 = np.float32(canvas_img2) / 255
    visualization_img1 = show_cam_on_image(src_img1, grayscale_cam1, use_rgb=False)
    visualization_img2 = show_cam_on_image(src_img2, grayscale_cam2, use_rgb=False)
    cv2.imshow('feature map_t1', visualization_img1)
    cv2.waitKey(0)
    cv2.imwrite(path_out + "feature_ad_t1.png", visualization_img1)
    cv2.imshow('feature map2_t2', visualization_img2)
    cv2.waitKey(0)
    cv2.imwrite(path_out + "feature_ad_t2.png", visualization_img2)


if __name__ == '__main__':
    # # print(get_model())
    # fp=open('./try_print_model.txt','a+')#是以读写的方式打开文件，若无文件则创建，若有则在原有文本上添加
    # print(get_model(), file=fp)#将内容输出到指定文件中
    # fp.close()#关闭文件

    path =  './DataVis/'
    vis(path_out = path)
