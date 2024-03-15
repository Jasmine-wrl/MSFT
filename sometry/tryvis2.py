import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

import utils
from argparse import ArgumentParser
from models.basic_model_vis import CDEvaluator


# from utils import GradCAM, show_cam_on_image
import pytorch_grad_cam 
from pytorch_grad_cam.utils.image import show_cam_on_image



######
def get_args():
    parser = ArgumentParser()

    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    ############################需要修改对应
    #########LEVIR
    parser.add_argument('--project_name', default='CD_base_transformer_pos_s4fpn_diff_dd8_e6d2_sk_LEVIR_b16_lr0.0001_adam_train_val_100_linear_nw4', type=str)  
    ##########DSIFN
    # parser.add_argument('--project_name', default='CD_base_transformer_pos_s5fpn_diff_dd8_e2d6_eca_DSIFN_b16_lr0.01_sgd_train_val_150_linear_nw4', type=str) 

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


def show_cam(mask: np.ndarray,
                    use_rgb: bool = False,
                    colormap: int = cv2.COLORMAP_JET,
                    image_weight: float = 0.5) -> np.ndarray:
    """ This function 产生 cam mask to an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        :returns: The default image with the cam overlay.
        """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def vis(path_out):

    def reshape_transform(tensor, height=16, width=16):
        # print(tensor.size())
        result = tensor.reshape([tensor.size(0),
        height, width, -1])
        
        # 将通道维度放到第一个位置
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # 1.定义模型结构，选取要可视化的层
    mymodel = get_model()
    # traget_layers = [resnet18.layer4[1].bn2]
    # ############LEVIR
    traget_layer1 = [mymodel.fpn]
    traget_layer2 = [mymodel.transformer] 
    traget_layer3 = [mymodel.transformer_decoder] 
    traget_layer4 = [mymodel.sk]
    traget_layer5 = [mymodel.diff] 

    # ###############DSIFN
    # traget_layer1 = [mymodel.fpn2]
    # traget_layer2 = [mymodel.transformer] 
    # traget_layer3 = [mymodel.transformer_decoder] 
    # traget_layer4 = [mymodel.eca]
    # traget_layer5 = [mymodel.diff] 

    # 2.读取图片，将图片转为RGB

    origin_img1 = cv2.imread('/home/wangruilan/CD/Datasets/LEVIR-CD-256-v1/A/test_40_1.png')
    origin_img2 = cv2.imread('/home/wangruilan/CD/Datasets/LEVIR-CD-256-v1/B/test_40_1.png')
    # origin_img1 = cv2.imread('/home/wangruilan/CD/Datasets/DSIFN-CD-256/A/17_4.png')
    # cv2.imwrite('./DataVis/After_Encoder/origin_img1.png', origin_img1)
    # origin_img2 = cv2.imread('/home/wangruilan/CD/Datasets/DSIFN-CD-256/B/17_4.png')
    # cv2.imwrite('./DataVis/After_Encoder/origin_img2.png', origin_img2)
    rgb_img1 = cv2.cvtColor(origin_img1, cv2.COLOR_BGR2RGB)
    rgb_img2 = cv2.cvtColor(origin_img2, cv2.COLOR_BGR2RGB)


    # 3.图片预处理：resize ToTensor()
    trans = transforms.ToTensor()
    crop_img1 = trans(rgb_img1)
    crop_img2 = trans(rgb_img2)

    net_input1 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(crop_img1).unsqueeze(0)
    net_input2 = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(crop_img2).unsqueeze(0)

    # net_input1 = (crop_img1).unsqueeze(0)
    # net_input2 = (crop_img2).unsqueeze(0)

    # 4.将裁剪后的Tensor格式的图像转为numpy格式，便于可视化
    canvas_img1 = (crop_img1*255).byte().numpy().transpose(1, 2, 0)
    canvas_img1 = cv2.cvtColor(canvas_img1, cv2.COLOR_RGB2BGR)
    canvas_img2 = (crop_img2*255).byte().numpy().transpose(1, 2, 0)
    canvas_img2 = cv2.cvtColor(canvas_img2, cv2.COLOR_RGB2BGR)
    # print(type(canvas_img2))
    # print(canvas_img2.shape)
    # print(canvas_img2)


    #5.实例化cam
    ##############  VIT部分需要 reshape_transform
    cam = pytorch_grad_cam.GradCAMPlusPlus(model=mymodel, target_layers=traget_layer3, reshape_transform = reshape_transform)

    # cam = pytorch_grad_cam.GradCAMPlusPlus(model=mymodel, target_layers=traget_layer1)
   
    
    grayscale_cam1, grayscale_cam2 = cam(input_tensor1=net_input1, input_tensor2=net_input2)
    grayscale_cam1 = grayscale_cam1[0, :]
    grayscale_cam2 = grayscale_cam2[0, :]


    # 6.将feature map与原图叠加并可视化
    src_img1 = np.float32(canvas_img1) / 255
    src_img2 = np.float32(canvas_img2) / 255
    # visualization_img1 = show_cam_on_image(src_img1, grayscale_cam1, use_rgb=True)
    # visualization_img2 = show_cam_on_image(src_img2, grayscale_cam2, use_rgb=True)
    
    visualization_img1 = show_cam_on_image(src_img1, grayscale_cam1, use_rgb=False)
    visualization_img2 = show_cam_on_image(src_img2, grayscale_cam2, use_rgb=False)
    
    heatmap_img1 = show_cam(mask = grayscale_cam1, use_rgb=False)
    heatmap_img2 = show_cam(mask = grayscale_cam2, use_rgb=False)
    # heatmap_img1 = show_cam(mask = grayscale_cam1, use_rgb=True)
    # heatmap_img2 = show_cam(mask = grayscale_cam2, use_rgb=True)
    # cv2.imwrite(path_out + "heatmap_adiff_t1.png", heatmap_img1)
    # cv2.imwrite(path_out + "heatmap_adiff_t2.png", heatmap_img2)
    cv2.imwrite(path_out + "test_40_1_heatmap_ad_t1.png", heatmap_img1)
    cv2.imwrite(path_out + "test_40_1_heatmap_ad_t2.png", heatmap_img2)

    # cv2.imshow('feature map_t1', visualization_img1)
    # cv2.waitKey(0)
    cv2.imwrite(path_out + "test_40_1_feature_ad_t1.png", visualization_img1)
    # cv2.imshow('feature map2_t2', visualization_img2)
    # cv2.waitKey(0)
    cv2.imwrite(path_out + "test_40_1_feature_ad_t2.png", visualization_img2)

if __name__ == '__main__':
    # print(get_model())
    # fp=open('./try_print_model.txt','a+')#是以读写的方式打开文件，若无文件则创建，若有则在原有文本上添加
    # print('DSIFN begin!!!!!!!!!!', file=fp)#将内容输出到指定文件中
    # print(get_model(), file=fp)#将内容输出到指定文件中
    # fp.close()#关闭文件

    # pathout =  './DataVis/LEVIR/After_FPN/'
    # pathout =  './DataVis/LEVIR/After_Encoder/'
    pathout =  './DataVis/LEVIR/After_Decoder/'
    # pathout =  './DataVis/LEVIR/After_Att/'
    # pathout =  './DataVis/LEVIR/After_Diff/'

    # pathout =  './DataVis/DSIFN/After_FPN/'
    # pathout =  './DataVis/DSIFN/After_Encoder/'
    # pathout =  './DataVis/DSIFN/After_Decoder/'
    # pathout =  './DataVis/DSIFN/After_Att/'
    # pathout =  './DataVis/DSIFN/After_Diff/'

    if not os.path.exists(pathout):
        os.makedirs(pathout)
    vis(path_out = pathout)
    print('yes!ok!')
