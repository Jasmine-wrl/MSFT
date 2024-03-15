import os
import numpy as np
from PIL import Image

# Predict = '图片路径'
# Predict = './color_images'
Predict = 'E:/WRL/Desktop/IDIP/results/proj2/CD_base_transformer_pos_s4fpn_diff_dd8_e2d6_LEVIR_b16_lr0.01_train_val_100_linear_nw4'
# Rootdict = '根路径'
# Rootdict = '../../Datasets/LEVIR-CD-256-v1'
Rootdict = 'E:/WRL/Desktop/IDIP/datasets/LEVIR-CD-256-v1/'

save_path = 'E:/WRL/Desktop/IDIP/results/color/CD_base_transformer_pos_s4fpn_diff_dd8_e2d6_LEVIR_b16_lr0.01_train_val_100_linear_nw4'


###############################################################################
# 基础函数
###############################################################################

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, 'label', img_name)


def get_changeimg_path(Predict,img_name):
    return os.path.join(Predict, img_name)


def color_label(img1, img2):
    w, h, _ = img1.shape
    # 需要重新赋值,因为图片只读
    img = np.array(img2)

    fp = np.array([255, 255, 255])
    fn = np.array([1, 1, 1])
    for i in range(0, w):
        for j in range(0, h):
            p1 = img1[i][j]
            p2 = img2[i][j]
            # false positive，根据自己需求修改对应颜色
            if ((p2 - p1) == fp).all():
                img[i][j] = [255, 0, 0]
            # false negative
            if ((p2 - p1) == fn).all():
                img[i][j] = [0, 0, 255]
    return img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(np.array(image_numpy, dtype=np.uint8))
    image_pil.save(image_path)

###############################################################################
# 颜色标注变化区域
###############################################################################

def change_pics():
    root_dir = Rootdict
    # list_path = os.path.join(root_dir, 'list', 'demo.txt')
    list_path = os.path.join(root_dir, 'list', 'test.txt')
    # save_path = os.path.join(root_dir, 'color_label', Predict)
 
    os.makedirs(save_path, exist_ok=True)
    img_name_list = load_img_name_list(list_path)
    size = len(img_name_list)
    for index in range(0, size):
        name = img_name_list[index]
        print('process:' + name)
        # A_path = get_img_path(root_dir, img_name_list[index % size])
        A_path = get_img_path(root_dir, name)

        # B_path = get_changeimg_path(root_dir, img_name_list[index % size])
        B_path = get_changeimg_path(Predict, name)
        a = Image.open(A_path)
        b = Image.open(B_path)

        #  灰度值转rgb
        img = np.asarray(a.convert('RGB'))
        img_B = np.asarray(b.convert('RGB'))

        #  颜色转化
        color_img = color_label(img, img_B)
        filename = os.path.join(save_path, name.replace('.jpg', '.png'))
        
        # 图片保存
        save_image(color_img, filename)

if __name__ == '__main__':
    change_pics()
