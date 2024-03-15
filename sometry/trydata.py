import numpy as np
import re
import os

################# G_loss 
def extract_data(filepath, keyword = 'Is_training: False', find = 'G_loss'):
    res = []
   
    with open(filepath, 'r') as file:
        for line in file.readlines():
            if keyword in line:
                match = re.search(find +':\s*(\d+\.\d+)', line) # 正则表达式匹配包含'loss:'及后跟小数点的行
                # match = re.search('G_loss:\s*(\d+\.\d+)', line) # 正则表达式匹配包含'loss:'及后跟小数点的行
                if match:
                    res.append(float(match.group(1))) # 如果匹配成功，将匹配到的值转换为浮点型并添加到数组中         
    return res

    
def extract_data_nextline(filepath, keyword = 'Is_training: False', find = 'mf1'):
    res = []
   
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            if keyword in lines[i]:
                line_new = lines[i+1]
                match = re.search(find +':\s*(\d+\.\d+)', line_new) # 正则表达式匹配包含'loss:'及后跟小数点的行
                # match = re.search('mf1:\s*(\d+\.\d+)', line_new) # 正则表达式匹配包含'loss:'及后跟小数点的行
                if match:
                    res.append(float(match.group(1))) # 如果匹配成功，将匹配到的值转换为浮点型并添加到数组中         
    return res

def save_data(path, data, name):
    output_file = open(path, 'a')
    output_file.write(name +': '+ str(data) + '\n')
    # 关闭文件
    output_file.close()
 
if __name__ == "__main__":
    # 调用函数进行测试
    path_root = '/home/wangruilan/CD/proj2/BIT_CD/checkpoints/CD_base_transformer_pos_s4fpn_diff_dd8_e2d6_DSIFN_b16_lr0.01_sgd_train_val_150_linear_nw4/'
    name_read = 'log.txt'
    path_read = path_root + name_read
    
    # path_write = './DataExtract/LEVIR/Lr/'
    path_write = './DataExtract/DSIFN/FPN&Diff/'
    name_write = 'ResViT_s4_fpn_diff.txt'
    # name_write = 'FC-Siam-Di.txt'
    # name_write = 'FC-Siam-Conc.txt'
    # name_write = 'DTCDSCN.txt'
    # name_write = 'BIT.txt'
    # name_write = 'ChangeFormer.txt'

    if not os.path.exists(path_write):
        os.makedirs(path_write)

    #################注意keyword的不同
    y_loss = extract_data(filepath= path_read, keyword= 'Is_training: False', find='G_loss')
    y_acc = extract_data_nextline(filepath= path_read, keyword= 'Is_training: False. Epoch', find='acc')
    y_f1 = extract_data_nextline(filepath= path_read, keyword= 'Is_training: False. Epoch', find='mf1')
    y_miou = extract_data_nextline(filepath= path_read, keyword= 'Is_training: False. Epoch', find='miou')
    y_precision_0 = extract_data_nextline(filepath= path_read, keyword= 'Is_training: False. Epoch', find='precision_0')
    y_precision_1 = extract_data_nextline(filepath= path_read, keyword= 'Is_training: False. Epoch', find='precision_1')
    y_recall_0 = extract_data_nextline(filepath= path_read, keyword= 'Is_training: False. Epoch', find='recall_0')
    y_recall_1 = extract_data_nextline(filepath= path_read, keyword= 'Is_training: False. Epoch', find='recall_1')
    

    output_file = open(path_write + name_write, 'a')
    output_file.write('loss: '+ str(y_loss) + '\n')
    output_file.write('acc: '+ str(y_acc) + '\n')
    output_file.write('f1: '+ str(y_f1) + '\n')
    output_file.write('miou: '+ str(y_miou) + '\n')
    output_file.write('precision_0: '+ str(y_precision_0) + '\n')
    output_file.write('precision_1: '+ str(y_precision_1) + '\n')
    output_file.write('recall_0: '+ str(y_recall_0) + '\n')
    output_file.write('recall_1: '+ str(y_recall_1) + '\n')

    # 关闭文件
    output_file.close()

    print("ok!!!!!")

    # save_data(path= path_write, data= y_loss, name= 'Loss')
   