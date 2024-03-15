import numpy as np

scores = np.load('./checkpoints/CD_base_transformer_pos_s4_dd8_dedim8_LEVIR_b16_lr0.01_train_val_100_linear/scores_dict.npy',allow_pickle=True)
print('scores:',scores)

val_acc = np.load('./checkpoints/CD_base_transformer_pos_s4_dd8_dedim8_LEVIR_b16_lr0.01_train_val_100_linear/val_acc.npy',allow_pickle=True)
print('val_acc', val_acc)