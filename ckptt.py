import torch
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger
import os
import glob
import numpy as np
import random
from torch.utils.data import Subset
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42) # 設定固定種子
# --- 設定區域 ---
root_log_dir = '/media/Siamese-Diffusion/lightning_logs' 
config_path = '/media/Siamese-Diffusion/models/cldm_v15.yaml'
base_save_dir = './batch_inference_results'

# 在這裡輸入您想跑的 version 編號
specified_versions = [151]

# 根據編號組成完整的路徑清單
all_versions = [os.path.join(root_log_dir, f'version_{v}') for v in specified_versions]

print(f"準備處理指定的 {len(all_versions)} 個版本。")

# --- 1. 初始化模型與數據 ---
print("正在初始化模型結構...")
model = create_model(config_path).cpu()

dataset = MyDataset()
# 固定一組測試資料， batch_size=4 會生成 4 張對比圖
subset_indices = [0, 1, 2, 3] # 指定你要測試的圖片編號
subset_dataset = Subset(dataset, subset_indices)
dataloader = DataLoader(subset_dataset, batch_size=4, shuffle=False)
test_batch = next(iter(dataloader))
# --- 2. 遍歷處理 ---
for v_path in all_versions:
    v_name = os.path.basename(v_path)
    
    # 檢查該路徑是否存在
    if not os.path.exists(v_path):
        print(f"錯誤：找不到路徑 {v_path}，跳過。")
        continue
    
    # 搜尋該版本下所有的 .ckpt 檔案
    ckpt_files = glob.glob(os.path.join(v_path, 'checkpoints/*.ckpt'))
    
    if not ckpt_files:
        print(f"[{v_name}] checkpoints 資料夾內找不到任何 .ckpt，跳過。")
        continue

    # 抓取該資料夾下「最後修改時間」最晚的 .ckpt
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    
    print(f"[{v_name}] 載入最新權重: {os.path.basename(latest_ckpt)}")

    try:
        # 載入權重並移動到 GPU
        model.load_state_dict(load_state_dict(latest_ckpt, location='cpu'), strict=False)
        model.cuda()
        model.eval()

        # 生成影像
        with torch.no_grad():
            images = model.log_images(test_batch, split=v_name)
            
            # --- 新增這一段：把所有 Tensor 轉到 CPU ---
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
            # ---------------------------------------

        # 存檔
        img_logger = ImageLogger(batch_frequency=1, rescale=True)
        img_logger.log_local(
            save_dir=base_save_dir,
            split=v_name,
            images=images,
            global_step=0, 
            current_epoch=0,
            batch_idx=0
        )
        
        # 釋放 GPU 記憶體防止 OOM
        model.cpu()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"處理 {v_name} 時發生錯誤: {e}")

print(f"\n任務完成！結果儲存在: {base_save_dir}/image_log/")