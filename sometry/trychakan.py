
# import torch

#     # Load category and color encodings
# cat_dict = torch.load('checkpoints/CD_base_transformer_pos_s4_dd8_dedim8_LEVIR_b4_lr0.01_train_val_100_linear/last_ckpt.pt',map_location=torch.device('cpu'))
# for k, v in cat_dict.items():  # k 参数名 v 对应参数值
#         print(k, v)


import torch
content = torch.load('checkpoints/CD_base_transformer_pos_s4_dd8_dedim8_LEVIR_b4_lr0.01_train_val_100_linear/last_ckpt.pt',map_location=torch.device('cpu') )
print(content.keys())   # keys()
# 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['model_G_state_dict'])

content2 = torch.load('checkpoints/seco_resnet18_1m.ckpt',map_location=torch.device('cpu') )
print(content2.keys())   # keys()