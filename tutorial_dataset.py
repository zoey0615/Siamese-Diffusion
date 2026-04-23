import cv2
import json
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import albumentations
import os

class MyDataset(Dataset):
    def __init__(self,json_path = '/media/Siamese-Diffusion/data/train_cvcv5.json'):
        self.data = [] #把 JSON 檔案裡每一行的內容讀進了 self.data 清單裡。
        self.data_root = '/media/Siamese-Diffusion/data'
        
        with open(json_path, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)
    def _local_enhance(self, image_np, mask_np):    #遮罩處理
        #色彩空間轉換
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) #原始影像有 Red, Green, Blue 三個通道 to 單通道的灰階圖
        mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        
        # 1.二值化+大津演算法
        _, binary_mask = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #忽略電腦自動計算出來的 最佳閾值
        
        # 2. 主動縮減管線寬度，使用 3x3 核心進行一次腐蝕，這會讓 5 像素寬的線變成 3 像素寬(還要更細，可以將 iterations 改為 2
        kernel = np.ones((3, 3), np.uint8)
        shrunk_mask = cv2.erode(binary_mask, kernel, iterations=1)
        affected_area = shrunk_mask.astype(float) / 255.0 #Alpha 通道（Alpha Map）」或「權重地圖)0-1矩陣
        
        # 3. [動態提亮]：維持組織感知邏輯，但稍微降低偏移量。讓導管亮得有層次，而不是死白一條
        brightness = gray.astype(float)
        dynamic_white = np.clip(brightness * 0.8 + 15, 0, 255).astype(np.uint8) #將背景亮度打 8 折+偏移量=導管
        
        # 4.[極致羽化]：將核心縮到最小 (3, 3)
        # 配合縮小後的 mask，這會產生極其纖細且邊緣微弱融合的效果
        alpha = cv2.GaussianBlur(affected_area, (3, 3), 0)
        
        # 5. 合成
        output_float = (dynamic_white.astype(float) * alpha + 
                        gray.astype(float) * (1.0 - alpha)) #(新管線亮度 * alpha) + (原始背景亮度 * (1.0 - alpha))
        
        return cv2.cvtColor(output_float.astype(np.uint8), cv2.COLOR_GRAY2RGB) #回傳一個三通道的 RGB 影像（Numpy Array）
    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = os.path.join(self.data_root, item['label'])
        target_filename = os.path.join(self.data_root, item['source'])
        prompt = item['prompt']

        p = random.random()
        if p > 0.95:
            prompt = ""

        #source = Image.open(source_filename).convert('L')
        #source_array = np.array(source)
        #threshold = 127
        #binary_array = np.where(source_array > threshold, 255, 0).astype(np.uint8)
        #binary_image = Image.fromarray(binary_array)
        #source = binary_image.convert('RGB')
        source = Image.open(source_filename).convert('RGB')
        target = Image.open(target_filename).convert('RGB')

        source_np = np.array(source).astype(np.uint8)
        target_np = np.array(target).astype(np.uint8)
        target_raw_np = target_np.copy()
        # 🔥 設定一個機率，例如 50% 的機率執行預繪製增強
        if random.random() < 0.5:
            target_np = self._local_enhance(target_np, source_np)
        else:
            # 另外 50% 保持乾淨，讓模型練習「從無到有」生成
            target_np = target_raw_np
        preprocess = self.transform(size=512)(image=target_np, mask=source_np)
        source, target = preprocess['mask'], preprocess['image']
        raw_preprocess = self.transform(size=512)(image=target_raw_np)
        target_raw = raw_preprocess['image']
        ############ Mask-Image Pair ############
        source = source.astype(np.float32) / 255.0
        target = target.astype(np.float32) / 127.5 - 1.0
        target_raw = target_raw.astype(np.float32) / 127.5 - 1.0
        return dict(jpg=target, txt=prompt, hint=source,filename=target_raw)

    def transform(self, size=512): #384 512
        transforms = albumentations.Compose(
                        [
                            albumentations.Resize(height=size, width=size)
                        ]
                    )
        return transforms





# import cv2
# import json
# import random
# import numpy as np
# import os
# from torch.utils.data import Dataset
# from PIL import Image
# import albumentations


# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = []
#         # 1. 這裡建議定義根目錄
#         self.data_root = '/media/Siamese-Diffusion/data' 
#         json_path = os.path.join(self.data_root, 'train_cvcv5.json') #'train_cvcv5.json'
        
#         with open(json_path, 'rt') as f:
#             for line in f:
#                 self.data.append(json.loads(line))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         item = self.data[idx]

#         # 2. 這裡要結合根目錄，不然會找不到檔案
#         source_filename = os.path.join(self.data_root, item['label'])
#         target_filename = os.path.join(self.data_root, item['source'])
#         prompt = item['prompt']

#         p = random.random()
#         if p > 0.95:
#             prompt = ""

#         source = Image.open(source_filename).convert('RGB')
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


        