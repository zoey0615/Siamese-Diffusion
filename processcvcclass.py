import os
import cv2
import pandas as pd
import numpy as np
import ast #Abstract Syntax Trees
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from scipy.interpolate import splprep, splev
import random

# ================= 配置設定 =================
CSV_PATH = '/media/Siamese-Diffusion/data/train_annotations.csv'
TRAIN_IMG_DIR = '/media/Siamese-Diffusion/data/train' 

# 這是主要的輸出根目錄
OUTPUT_ROOT = '/media/Siamese-Diffusion/data'

TARGET_SIZE = (512, 512)
LINE_THICKNESS = 4
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
        for sub_dir in ['images_thickvv5', 'masks_thickvv5']:
            path = os.path.join(OUTPUT_ROOT, split, sub_dir)
            os.makedirs(path, exist_ok=True)
    
    print(f"📁 已初始化資料集目錄結構: {OUTPUT_ROOT}")
    
    # 讀取原始資料
    df = pd.read_csv(CSV_PATH)
    total_raw = len(df['StudyInstanceUID'].unique())
    print(f"   [Step 1] 原始標註影像總數: {total_raw} 張")

    # [Step 2] 篩選包含 CVC 的標註
    cvc_df = df[df['label'].str.contains('CVC', na=False)].reset_index(drop=True)
    
    # [Step 3] 嚴格篩選單根管線 (核心修正處)
    # ✨ 關鍵：必須先問原始 df 誰才是真正的單管影像，避免雜質被過濾後造成誤判
    original_counts = df.groupby('StudyInstanceUID').size()
    true_single_ids = original_counts[original_counts == 1].index.tolist()
    
    # 建立分層抽樣用的 DataFrame：必須在「真的只有一根管子」且「那根是 CVC」的名單內
    stratify_df = cvc_df[cvc_df['StudyInstanceUID'].isin(true_single_ids)][['StudyInstanceUID', 'label']]
    stratify_df = stratify_df.drop_duplicates('StudyInstanceUID')
    
    print(f"   [Step 3] 嚴格排除多重/雜質管線後 (僅保留純淨單根 CVC): {len(stratify_df)} 張")
    print("           各類別原始分佈：")
    print(stratify_df['label'].value_counts())
    
    # [Step 4] 執行分層資料集分割
    try:
        train_ids_arr, val_ids_arr = train_test_split(
            stratify_df['StudyInstanceUID'].values,
            test_size=SPLIT_RATIO, 
            random_state=42,
            stratify=stratify_df['label'].values
        )
    except ValueError as e:
        print(f"❌ 分層抽樣失敗: {e}")
        return

    train_ids = set(train_ids_arr)
    val_ids = set(val_ids_arr)
    
    print(f"   [Step 4] 分層分割完成。訓練集: {len(train_ids)} 張, 驗證集: {len(val_ids)} 張")

    train_entries = []
    val_entries = []

    # [Step 5] 開始批次處理
    for _, info_row in tqdm(stratify_df.iterrows(), total=len(stratify_df), desc="處理進度"):
        image_id = info_row['StudyInstanceUID']
        current_label = info_row['label']
        
        src_img_path = os.path.join(TRAIN_IMG_DIR, f"{image_id}.jpg")
        if not os.path.exists(src_img_path): continue
        
        orig_img = cv2.imread(src_img_path)
        if orig_img is None: continue 
        orig_h, orig_w = orig_img.shape[:2]
        resized_img = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        mask = np.zeros(TARGET_SIZE, dtype=np.uint8)
        
        # 取得標註數據
        row = cvc_df[cvc_df['StudyInstanceUID'] == image_id].iloc[0]
        try:
            if image_id in train_ids:
                split_tag, current_list = "train", train_entries
            else:
                split_tag, current_list = "val", val_entries

            prompt_text = (
                f"High-quality clinical chest x-ray, clear thoracic anatomy with visible rib cage and spine, "
                f"distinct cardiac silhouette and heart borders, "
                f"a single central venous catheter positioned correctly in the mediastinum, {current_label}"
            )          
            
            # 繪製邏輯
            raw_points = ast.literal_eval(row['data'])
            points = np.array(raw_points, dtype=np.float32)
            points[:, 0] *= (TARGET_SIZE[0] / orig_w)
            points[:, 1] *= (TARGET_SIZE[1] / orig_h)

            # B-spline 插值
            if len(points) > 3:
                tck, u = splprep([points[:, 0], points[:, 1]], s=0)
                u_fine = np.linspace(0, 1, 200)
                x_fine, y_fine = splev(u_fine, tck)
                pts_smooth = np.vstack((x_fine, y_fine)).T.astype(np.int32)
            else:
                pts_smooth = points.astype(np.int32)

            cv2.polylines(mask, [pts_smooth], isClosed=False, color=255, 
                         thickness=LINE_THICKNESS, lineType=cv2.LINE_AA)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.0)

            # 存檔
            img_save_name = f"{image_id}.jpg"
            img_save_path = os.path.join(OUTPUT_ROOT, split_tag, "images_thickvv5", img_save_name)
            cv2.imwrite(img_save_path, resized_img)

            mask_save_name = f"{image_id}.png"
            mask_save_path = os.path.join(OUTPUT_ROOT, split_tag, "masks_thickvv5", mask_save_name)
            cv2.imwrite(mask_save_path, mask)

            current_list.append({
                "source": f"{split_tag}/images_thickvv5/{img_save_name}",
                "label": f"{split_tag}/masks_thickvv5/{mask_save_name}",
                "prompt": prompt_text,
                "name": image_id
            })

        except Exception as e:
            print(f"\n❌ 錯誤影像 ID: {image_id} | 原因: {e}")
            continue

    # [Step 6] 執行下取樣 (針對 Normal)
    balanced_train_entries = []
    remaining_normal_entries = [] 
    NORMAL_KEEP_RATIO = 0.4 
    random.seed(42) 

    count_dict = {"Normal": 0, "Borderline": 0, "Abnormal": 0}
    
    for entry in train_entries:
        if "Abnormal" in entry['prompt']:
            balanced_train_entries.append(entry)
            count_dict["Abnormal"] += 1
        elif "Borderline" in entry['prompt']:
            balanced_train_entries.append(entry)
            count_dict["Borderline"] += 1
        else:
            if random.random() < NORMAL_KEEP_RATIO:
                balanced_train_entries.append(entry)
                count_dict["Normal"] += 1
            else:
                remaining_normal_entries.append(entry)

    # 提取推論底圖
    HEALTH_OUT_DIR = os.path.join(OUTPUT_ROOT, 'healthy_xray1')
    os.makedirs(HEALTH_OUT_DIR, exist_ok=True)
    
    NUM_INFERENCE_BG = 1000
    inference_samples = random.sample(
        remaining_normal_entries, 
        min(NUM_INFERENCE_BG, len(remaining_normal_entries))
    )

    print(f"📦 正在提取 {len(inference_samples)} 張推論底圖...")
    for sample in tqdm(inference_samples, desc="儲存底圖"):
        src_path = os.path.join(OUTPUT_ROOT, sample['source'])
        save_path = os.path.join(HEALTH_OUT_DIR, os.path.basename(src_path).replace('.jpg', '.png'))
        img = cv2.imread(src_path)
        if img is not None:
            cv2.imwrite(save_path, img)

    print(f"\n📊 平衡結果: Normal: {count_dict['Normal']}, Borderline: {count_dict['Borderline']}, Abnormal: {count_dict['Abnormal']}")
    
    save_json_lines(balanced_train_entries, os.path.join(OUTPUT_ROOT, "train_cvcvv5.json"))
    save_json_lines(val_entries, os.path.join(OUTPUT_ROOT, "val_cvcvv5.json"))

    print(f"\n✨ 任務完成！驗證集現在應該非常純淨了。")

if __name__ == "__main__":
    generate_masks()