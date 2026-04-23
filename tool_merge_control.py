# 1. （這是 SD 1.5 的基礎權重）
path_sd15 = './models/v1-5-pruned1.ckpt' 

# 2. 指向你實際生成的 12GB 完整權重 (version)ControlNet 模型權重
path_sd15_with_control = './lightning_logs/version_152/checkpoints/epoch=9-step=5000.ckpt'

# 3. 定義你要存檔的位置與名稱
path_output = './merged_model_version152.ckpt'


import os


assert os.path.exists(path_sd15), 'Input path_sd15 does not exists!'
assert os.path.exists(path_sd15_with_control), 'Input path_sd15_with_control does not exists!'
assert os.path.exists(os.path.dirname(path_output)), 'Output folder not exists!'


import torch
from share import *
from cldm.model import load_state_dict


sd15_state_dict = load_state_dict(path_sd15)
sd15_with_control_state_dict = load_state_dict(path_sd15_with_control)


# Merge state dicts
final_state_dict = sd15_state_dict.copy()  # Start with sd15 weights

for key, value in sd15_with_control_state_dict.items():
    print(f"Overriding {key} with sd15_with_control weights")
    final_state_dict[key] = value  # Override with sd15_with_control weights if exists

torch.save(final_state_dict, path_output)
print('Merged model saved at ' + path_output)
