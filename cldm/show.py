import os
import random
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torchvision.utils import make_grid

def create_scientific_comparison(folder_dict, num_samples=5, img_size=256, output_path="comparison_rows.png", specified_files=None):
    """
    folder_dict: {"實驗名稱": "影像資料夾路徑"}
    num_samples: 你要比幾組 (每一組會產生 2 列：影像列 + 遮罩列)
    """
    all_tensors = []
    labels = list(folder_dict.keys())
    
    # 1. 取得基準檔名 (以第一個資料夾為準)
    if specified_files is not None:
        # 使用你指定的檔名
        selected_files = specified_files
        num_samples = len(selected_files)
    else:
        # 原本的隨機抽樣邏輯
        first_folder = list(folder_dict.values())[0]
        filenames = sorted([f for f in os.listdir(first_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        num_samples = min(len(filenames), num_samples)
        selected_files = random.sample(filenames, num_samples)

    # 2. 核心邏輯：依照「組」來循環
    for i, file_name in enumerate(selected_files):
        
        # --- 第一行：所有實驗的影像 (橫向排滿 5 欄) ---
        for label, img_folder in folder_dict.items():
            img_path = os.path.join(img_folder, file_name)
            img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
            all_tensors.append(T.ToTensor()(img))
            
        # --- 第二行：所有實驗的遮罩 (橫向排滿 5 欄) ---
        for label, img_folder in folder_dict.items():
            # 假設遮罩在 images 資料夾旁的 masks 資料夾內
            mask_folder = img_folder.replace("images", "masks") 
            mask_path = os.path.join(mask_folder, file_name)
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('RGB').resize((img_size, img_size))
            else:
                # 找不到就顯示全黑
                mask = Image.new('RGB', (img_size, img_size), (0, 0, 0)) 
            all_tensors.append(T.ToTensor()(mask))

    # 3. 建立網格
    # nrow 設為實驗數量 (如果你輸入 5 個資料夾，nrow 就是 5)
    grid = make_grid(torch.stack(all_tensors), nrow=len(labels), padding=10)
    result_img = T.ToPILImage()(grid)

    # 4. 繪製頂部實驗標籤
    draw = ImageDraw.Draw(result_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        # 計算每欄 X 座標
        x_pos = 10 + i * (img_size + 10) + (img_size // 4)
        draw.text((x_pos, 2), label, fill=(255, 0, 0), font=font)

    # 5. 儲存
    result_img.save(output_path)
    print(f"✅ 對比圖已生成：{output_path}")
    print(f"佈局：{len(labels)} 欄 x {num_samples * 2} 列")


base_path ="/media//Siamese-Diffusion/generated_results"
# --- 使用範例：輸入你的 5 個實驗路徑 ---
my_experiments = {
    "version31": os.path.join(base_path, "version_31/images"),
    "version32": os.path.join(base_path, "version_32/images"),
    "version34": os.path.join(base_path, "version_34/images"),
    "version35": os.path.join(base_path, "version_35/images"),
    "version36": os.path.join(base_path, "version_36/images"),
}
my_target_files = ["b-001372_idx-0.png", "b-001388_idx-0.png", "b-001424_idx-0.png","b-000004_idx-0.png","b-001042_idx-0.png"]
# 執行腳本 (會生成 5 欄 x 10 列的大圖)
create_scientific_comparison(
    my_experiments, 
    img_size=384, 
    specified_files=my_target_files, # <--- 這裡指定！
    output_path=os.path.join(base_path, "thesis_comparison_v2.png")
)
