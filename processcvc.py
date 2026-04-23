import os
import cv2
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from scipy.interpolate import splprep, splev
# ================= 配置設定 =================
CSV_PATH = '/media/Siamese-Diffusion/data/train_annotations.csv'
TRAIN_IMG_DIR = '/media/Siamese-Diffusion/data/train' 

# 這是主要的輸出根目錄
OUTPUT_ROOT = '/media/Siamese-Diffusion/data'

TARGET_SIZE = (384, 384)
LINE_THICKNESS = 6
SPLIT_RATIO = 0.2
# ===========================================

def save_json_lines(data, path):
    """將 list 儲存為 JSON Lines 格式"""
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def generate_masks():
    # [Step 0] 建立結構化的目錄
    for split in ['train', 'val']:
        for sub_dir in ['images_thick6_new', 'masks_thick6_new']:
            path = os.path.join(OUTPUT_ROOT, split, sub_dir)
            os.makedirs(path, exist_ok=True)
    
    print(f"📁 已初始化資料集目錄結構: {OUTPUT_ROOT}")
    print("🚀 開始資料預處理與實體切分流程...")
    
    # 讀取原始資料
    df = pd.read_csv(CSV_PATH)
    total_raw = len(df['StudyInstanceUID'].unique())
    print(f"  [Step 1] 原始標註影像總數: {total_raw} 張")

    # [Step 2] 篩選包含 CVC 的標註
    cvc_df = df[df['label'].str.contains('CVC', na=False)].reset_index(drop=True)
    cvc_image_count = len(cvc_df['StudyInstanceUID'].unique())
    print(f"  [Step 2] 含有 CVC 標註的影像: {cvc_image_count} 張")
    
    # [Step 3] 篩選單根管線
    counts = cvc_df.groupby('StudyInstanceUID').size()
    single_cvc_ids = counts[counts == 1].index.tolist()
    cvc_df = cvc_df[cvc_df['StudyInstanceUID'].isin(single_cvc_ids)].reset_index(drop=True)
    print(f"  [Step 3] 排除多重管線後 (僅保留單根 CVC): {len(single_cvc_ids)} 張")
    
    # [Step 4] 執行資料集分割 (8:2)
    train_ids, val_ids = train_test_split(
        single_cvc_ids, 
        test_size=SPLIT_RATIO, 
        random_state=42 
    )
    
    print(f"  [Step 4] 分割完成。預計訓練集: {len(train_ids)} 張, 驗證集: {len(val_ids)} 張")

    train_entries = []
    val_entries = []

    # [Step 5] 開始批次處理
    for image_id in tqdm(single_cvc_ids, desc="處理進度"):
        # ... (讀取與 Resize 影像部分保持不變) ...
        src_img_path = os.path.join(TRAIN_IMG_DIR, f"{image_id}.jpg")
        if not os.path.exists(src_img_path): continue
        orig_img = cv2.imread(src_img_path)
        if orig_img is None: continue 
        orig_h, orig_w = orig_img.shape[:2]
        resized_img = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
        row = cvc_df[cvc_df['StudyInstanceUID'] == image_id].iloc[0]

        try:
            # (判斷分類標籤與 Prompt 邏輯保持不變)
            if image_id in train_ids:
                split_tag, current_list = "train", train_entries
            else:
                split_tag, current_list = "val", val_entries

            prompt_text = "high-resolution chest x-ray, single central venous catheter "

            # --- 優化後的繪製邏輯 ---
            raw_points = ast.literal_eval(row['data'])
            points = np.array(raw_points, dtype=np.float32)
            # 座標縮放
            points[:, 0] *= (TARGET_SIZE[0] / orig_w)
            points[:, 1] *= (TARGET_SIZE[1] / orig_h)

            # 1. 使用 B-spline 進行平滑插值
            if len(points) > 3:
                # s=0 表示曲線必須經過所有點，增加線條流暢度
                tck, u = splprep([points[:, 0], points[:, 1]], s=0)
                u_fine = np.linspace(0, 1, 200) # 均勻插入 200 個點
                x_fine, y_fine = splev(u_fine, tck)
                pts_smooth = np.vstack((x_fine, y_fine)).T.astype(np.int32)
            else:
                pts_smooth = points.astype(np.int32)

            # 2. 繪製平滑曲線 (取消 shift=8 改用插值點直接畫)
            cv2.polylines(
                mask, 
                [pts_smooth], 
                isClosed=False, 
                color=255, 
                thickness=LINE_THICKNESS, 
                lineType=cv2.LINE_AA  # 抗鋸齒
            )

            # 3. 形態學處理：確保像素完全閉合，消除肉眼看不見的細微斷點
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 4. 適度的高斯模糊，讓擴散模型更容易理解邊緣梯度
            mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.0)
            # --- 實體存檔 ---
            img_save_name = f"{image_id}.jpg"
            img_save_path = os.path.join(OUTPUT_ROOT, split_tag, "images_thick6_new", img_save_name)
            cv2.imwrite(img_save_path, resized_img)

            mask_save_name = f"{image_id}.png"
            mask_save_path = os.path.join(OUTPUT_ROOT, split_tag, "masks_thick6_new", mask_save_name)
            cv2.imwrite(mask_save_path, mask)

            # 紀錄 JSON
            current_list.append({
                "source": f"{split_tag}/images_thick6_new/{img_save_name}",
                "label": f"{split_tag}/masks_thick6_new/{mask_save_name}",
                "prompt": prompt_text,
                "name": image_id
            })

        except Exception:
            continue

    # [Step 6] 儲存 JSON 檔案
    save_json_lines(train_entries, os.path.join(OUTPUT_ROOT, "train_cvc_new6.json"))
    save_json_lines(val_entries, os.path.join(OUTPUT_ROOT, "val_cvc_new6.json"))

    print(f"\n✨ 任務完成！所有資料已切分並存儲於: {OUTPUT_ROOT}")
    print(f"📊 最終成功產出：訓練集 {len(train_entries)} 筆, 驗證集 {len(val_entries)} 筆")

if __name__ == "__main__":
    generate_masks()