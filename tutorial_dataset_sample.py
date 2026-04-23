import cv2
import json
import random
import numpy as np
import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import albumentations


class MyDataset(Dataset):
    def __init__(self, mode='test'):
        self.data = []
        self.base_path = '/media/Siamese-Diffusion'
        self.data_root = os.path.join(self.base_path, 'data')

        json_name = 'test_cvcv5_clean.json'
        json_path = os.path.join(self.data_root, json_name)
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到 JSON 檔案: {json_path}")
            
        with open(json_path, 'rt') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        # 2. 建立底圖庫 (1,000 張真實 X 光片)
        bg_folder = os.path.join(self.data_root, 'healthy_xrays') 
        self.bg_paths = glob.glob(os.path.join(bg_folder, "*.png"))
        
        if len(self.bg_paths) == 0:
            print("⚠️ 警告：找不到健康底圖庫，將退回使用 JSON 內的原始底圖。")

    def __len__(self):
        return len(self.data)

    def _local_enhance(self, image_np, mask_np):
        """
        🎯 [訓練/推論一致版] 
        完全複刻訓練時的邏輯：細化管線 + 動態提亮 + 極致羽化
        """
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 處理 Mask 轉單通道
        if len(mask_np.shape) == 3:
            mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        else:
            mask_gray = mask_np
            
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
        
        # 🔥 [核心 1：細化管線] 讓 5 像素寬變細，與訓練集標註一致
        #kernel = np.ones((3, 3), np.uint8)
        #shrunk_mask = cv2.erode(binary_mask, kernel, iterations=1)
        kernel_enhance = np.ones((3, 3), np.uint8)
        strong_mask = cv2.dilate(binary_mask, kernel_enhance, iterations=1)
        affected_area = strong_mask.astype(float) / 255.0
        
        # 🔥 [核心 2：動態提亮] 維持組織感知，非死白
        brightness = gray.astype(float)
        dynamic_white = np.clip(brightness * 0.8 + 15, 0, 255).astype(np.uint8)
        
        # 🔥 [核心 3：極致羽化] 最小化模糊範圍 (3, 3)
        alpha = affected_area
        
        # 合成
        output_float = (dynamic_white.astype(float) * alpha + 
                        gray.astype(float) * (1.0 - alpha))
        
        return cv2.cvtColor(output_float.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    def __getitem__(self, idx):
        item = self.data[idx]
        # 📍 1. 取得新管線的遮罩路徑 (妳要生成的目標位置)
        source_filename = os.path.join(self.base_path, item['source']) 
        
        # 🖼️ 2. 隨機抽樣底圖 (路徑在 data/healthy_xrays)
        if self.bg_paths:
            target_filename = random.choice(self.bg_paths)
            
            # --- ✨ [影像修補：讓原本的管線消失] ---
            # 建立對應的舊遮罩路徑 (data/healthy_xrays -> data/mask/masks_thickv5)
            # 注意：這裡根據妳提供的路徑結構進行替換
            bg_mask_path = target_filename.replace('healthy_xrays', 'mask/masks_thickv5')
            
            # 讀取底圖 (使用 OpenCV)
            target_cv2 = cv2.imread(target_filename)
            
            if os.path.exists(bg_mask_path):
                # 讀取舊管線的標註遮罩 (灰階)
                old_tube_mask = cv2.imread(bg_mask_path, 0)
                
                # 確保遮罩與底圖尺寸完全一致
                if old_tube_mask.shape[:2] != target_cv2.shape[:2]:
                    old_tube_mask = cv2.resize(old_tube_mask, (target_cv2.shape[1], target_cv2.shape[0]))
                
                kernel_dilate = np.ones((5, 5), np.uint8) 
                old_tube_mask_fat = cv2.dilate(old_tube_mask, kernel_dilate, iterations=1)
                
                # 🔥 【關鍵修改 2】加大修補半徑
                # 將 inpaintRadius 從 3 提高到 5，讓過渡更自然
                target_np = cv2.inpaint(target_cv2, old_tube_mask_fat, 3, cv2.INPAINT_TELEA)
                target_np = cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB)
            else:
                # 如果找不到舊遮罩，則直接讀取原圖並轉換
                target_np = cv2.cvtColor(target_cv2, cv2.COLOR_BGR2RGB)
            # ----------------------------------------
        else:
            # 如果沒有底圖庫，退回使用 JSON 內的原始底圖
            target_filename = os.path.join(self.base_path, item['target'])
            target_np = np.array(Image.open(target_filename).convert('RGB'))

        # 📍 3. 讀取妳「新」要生成的導管遮罩
        source = Image.open(source_filename).convert('RGB')
        source_np = np.array(source).astype(np.uint8)

        # ✨ 4. 執行本地增強 (將新管子「融合」進修補後的乾淨底圖)
        # 即使底圖修補處有些微模糊，_local_enhance 會用新的高品質特徵覆蓋上去
        target_np = self._local_enhance(target_np, source_np)

        # 📍 5. Resize 與 歸一化
        preprocess = self.transform(size=512)(image=target_np, mask=source_np)
        source_res, target_res = preprocess['mask'], preprocess['image']

        # hint 供 ControlNet Mask 分支使用 (0~1)
        source_final = source_res.astype(np.float32) / 255.0
        
        # jpg 供 UNet 主模型使用 (-1~1)
        target_final = target_res.astype(np.float32) / 127.5 - 1.0

        # 🔥【關鍵修改】c_concat_image 應該是 0~1 的格式，方便模型做融合運算
        # 這樣 ControlLDM 裡的 get_input 讀取時才不會數值混亂
        control_image_final = target_res.astype(np.float32) / 255.0

        return dict(
            jpg=target_final,          # UNet 訓練目標 (-1~1)
            txt=item['prompt'], 
            hint=source_final,        # Mask 控制輸入 (0~1)
            
            # 🔥 這裡統一命名，確保 ControlLDM 能抓到正確的「底圖」
            c_concat_image=control_image_final, 
            
            # 為了相容性，可以多留這幾個，但數值都要統一成 0~1
            control_img=control_image_final,
            image=control_image_final,
            
            # 紀錄資訊用
            filename=item.get('filename', ''), # 妳的 DEBUG 顯示有這個 key
            orig_mask_path=source_filename,
            bg_path=target_filename
        )

    def transform(self, size=512): 
        return albumentations.Compose([
            albumentations.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR)
        ])




# #沒底圖
# import cv2
# import json
# import random
# import numpy as np

# from torch.utils.data import Dataset
# from PIL import Image
# import albumentations


# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         root = './data/test_cvcv5_clean.json'
#         with open(root, 'rt') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         source_filename = item['source']
#         target_filename = item['target']
#         prompt = item['prompt']

#         source = Image.open(source_filename).convert('L')
#         source_array = np.array(source)
#         threshold = 127
#         binary_array = np.where(source_array > threshold, 255, 0).astype(np.uint8)
#         binary_image = Image.fromarray(binary_array)  
#         source = binary_image.convert('RGB')

#         target = Image.open(target_filename).convert('RGB')

#         source = np.array(source).astype(np.uint8)
#         target = np.array(target).astype(np.uint8)

#         preprocess = self.transform()(image=target, mask=source)
#         source, target = preprocess['mask'], preprocess['image']

#         ############ Mask-Image Pair ############
#         source = source.astype(np.float32) / 255.0
#         target = target.astype(np.float32) / 127.5 - 1.0

#         return dict(jpg=target, txt=prompt, hint=source)


#     def transform(self, size=512):
#         transforms = albumentations.Compose(
#                         [
#                             albumentations.Resize(height=size, width=size)
                                                            
#                         ]
#                     )
#         return transforms
    