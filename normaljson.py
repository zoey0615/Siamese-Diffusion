import os
import cv2
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import json
from scipy.interpolate import splprep, splev

# ================= 配置設定 =================
CSV_PATH = '/media/Siamese-Diffusion/data/train_annotations.csv'
TRAIN_IMG_DIR = '/media/Siamese-Diffusion/data/train' 
OUTPUT_ROOT = '/media/Siamese-Diffusion/data'

# 輸出子目錄：妳的 Mask 會存在這裡
SAVE_SUBDIR = "reprocessed_v5_final"
TARGET_SIZE = (512, 512)
LINE_THICKNESS = 4

# 推論時使用的黑圖佔位符路徑
DUMMY_BLACK_PATH = "data/test_dummy_black.png"
# ===========================================

def save_json_lines(data, path):
    """儲存為 JSON Lines 格式"""
    with open(path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def generate_inference_json():
    # 1. 初始化目錄
    img_dir = os.path.join(OUTPUT_ROOT, SAVE_SUBDIR, "images")
    mask_dir = os.path.join(OUTPUT_ROOT, SAVE_SUBDIR, "masks")
    json_out_dir = os.path.join(OUTPUT_ROOT, "class_specific_json_v5")
    
    for d in [img_dir, mask_dir, json_out_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. 讀取與篩選
    df = pd.read_csv(CSV_PATH)
    uid_counts = df.groupby('StudyInstanceUID').size()
    single_pipe_ids = uid_counts[uid_counts == 1].index
    cvc_df = df[(df['StudyInstanceUID'].isin(single_pipe_ids)) & 
                (df['label'].str.contains('CVC', na=False))].copy()

    print(f"🎯 篩選完成！準備生成推論用格式，總數: {len(cvc_df)} 張")

    # 分類容器
    class_data = {
        'Normal': [],
        'Abnormal': [],
        'Borderline': []
    }

    # 3. 開始重繪與封裝
    for _, row in tqdm(cvc_df.iterrows(), total=len(cvc_df), desc="處理進度"):
        image_id = row['StudyInstanceUID']
        current_label = row['label']
        
        src_img_path = os.path.join(TRAIN_IMG_DIR, f"{image_id}.jpg")
        if not os.path.exists(src_img_path): continue
        
        orig_img = cv2.imread(src_img_path)
        if orig_img is None: continue
        orig_h, orig_w = orig_img.shape[:2]
        
        # 影像縮放 (這步是為了取得座標縮放比例，並存一份 512 的底圖備用)
        resized_img = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # --- 核心重畫 Mask 邏輯 ---
        mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
        try:
            raw_points = ast.literal_eval(row['data'])
            points = np.array(raw_points, dtype=np.float32)
            
            # 座標映射至 512x512
            points[:, 0] *= (TARGET_SIZE[0] / orig_w)
            points[:, 1] *= (TARGET_SIZE[1] / orig_h)

            # B-spline 平滑
            if len(points) > 3:
                tck, u = splprep([points[:, 0], points[:, 1]], s=0)
                u_fine = np.linspace(0, 1, 200)
                x_fine, y_fine = splev(u_fine, tck)
                pts_smooth = np.vstack((x_fine, y_fine)).T.astype(np.int32)
            else:
                pts_smooth = points.astype(np.int32)

            # 繪圖與後處理
            cv2.polylines(mask, [pts_smooth], isClosed=False, color=255, 
                         thickness=LINE_THICKNESS, lineType=cv2.LINE_AA)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.0)
            
            # 存檔
            img_name = f"{image_id}.jpg"
            mask_name = f"{image_id}.png"
            # 存下縮小的底圖與新畫的 Mask
            cv2.imwrite(os.path.join(img_dir, img_name), resized_img)
            cv2.imwrite(os.path.join(mask_dir, mask_name), mask)

            # 🔥 符合推論格式的封裝 🔥
            entry = {
                "source": f"data/{SAVE_SUBDIR}/masks/{mask_name}", # 輸入條件變成 Mask
                "target": DUMMY_BLACK_PATH,                         # 固定指向黑圖佔位符
                "prompt": (
                    f"High-quality clinical chest x-ray, clear thoracic anatomy with visible rib cage and spine, "
                    f"distinct cardiac silhouette and heart borders, "
                    f"a single central venous catheter positioned correctly in the mediastinum, {current_label}"
                )
            }

            # 按 Label 歸類
            if 'Normal' in current_label:
                class_data['Normal'].append(entry)
            elif 'Abnormal' in current_label:
                class_data['Abnormal'].append(entry)
            elif 'Borderline' in current_label:
                class_data['Borderline'].append(entry)

        except Exception as e:
            continue

    # 4. 輸出三個分類的 JSON 檔
    print("\n" + "="*30)
    for cls_name, entries in class_data.items():
        out_filename = f"cvc_{cls_name.lower()}_inference.json"
        out_path = os.path.join(json_out_dir, out_filename)
        save_json_lines(entries, out_path)
        print(f"✅ {cls_name:10}: {len(entries):5} 筆 -> {out_filename}")
    print("="*30)

if __name__ == "__main__":
    generate_inference_json()