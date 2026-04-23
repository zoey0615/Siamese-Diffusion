import os
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import torchvision.transforms as T
from torchvision.utils import make_grid

def create_scientific_contour_comparison(folder_dict, img_size=512, output_path="comparison_overlay.png", specified_files=None):
    """
    將遮罩轉換為淡黃色輪廓並疊加在影像上，以利觀察導管生成品質。
    """
    all_tensors = []
    labels = list(folder_dict.keys())
    
    if specified_files is None:
        print("❌ 請指定檔名列表 (specified_files)！")
        return
    
    selected_files = specified_files

    # 核心邏輯：每一組樣本佔一列
    for file_name in selected_files:
        # 橫向遍歷所有實驗版本
        for label, img_folder in folder_dict.items():
            img_path = os.path.join(img_folder, file_name)
            # 根據妳目前的目錄結構：images 資料夾對應 original_copied_masks
            mask_folder = img_folder.replace("images", "original_copied_masks") 
            mask_path = os.path.join(mask_folder, file_name)
            
            # 1. 讀取底圖 (X-ray 影像)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
            else:
                img = Image.new('RGB', (img_size, img_size), (50, 50, 50)) # 找不到顯示深灰
            
            # 2. 處理遮罩並提取「淡黃色輪廓」
# 2. 處理遮罩並提取「細緻外擴輪廓」
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L').resize((img_size, img_size))
                mask = mask.point(lambda x: 255 if x > 128 else 0)
                
                # --- 核心修改：兩層濾波法 ---
                # 第一層：決定輪廓的「外緣」位置 (數字越大，離導管越遠/看起來越寬)
                outer_edge = mask.filter(ImageFilter.MaxFilter(9)) 
                
                # 第二層：決定輪廓的「內緣」位置 (與外緣差值越小，線條越細)
                # 外緣設 7，內緣設 5，代表線條寬度約為 2 像素
                inner_edge = mask.filter(ImageFilter.MaxFilter(7)) 
                
                # 相減得到精細的環狀線
                edge = ImageChops.subtract(outer_edge, inner_edge)
                
                # 建立淡黃色層
                yellow_layer = Image.new("RGB", (img_size, img_size), (255, 255, 150))
                
                # 貼上
                img.paste(yellow_layer, (0, 0), edge)
                overlay_img = img
            else:
                overlay_img = img # 沒遮罩就顯示原圖
            
            all_tensors.append(T.ToTensor()(overlay_img))

    # 3. 建立網格 (nrow 設為實驗版本數量)
    grid = make_grid(torch.stack(all_tensors), nrow=len(labels), padding=10, pad_value=1.0) # 白色邊框
    result_img = T.ToPILImage()(grid)

    # 4. 繪製頂部標籤
    draw = ImageDraw.Draw(result_img)
    try:
        # 如果妳有 arial.ttf 可以調整字體大小，否則使用預設
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    for i, label in enumerate(labels):
        # 計算每欄 X 座標並稍微置中
        x_pos = 10 + i * (img_size + 10) + 20
        # 標籤也改為淡黃色或黃色以保持一致
        draw.text((x_pos, 5), label, fill=(255, 255, 0), font=font)

    # 5. 儲存結果
    result_img.save(output_path)
    print(f"✅ 淡黃色輪廓對比圖已生成：{output_path}")
    print(f"佈局規模：{len(labels)} 欄 x {len(selected_files)} 列")

# --- 執行部分 ---
base_path = "/media/Siamese-Diffusion/generated_results"

# 妳可以隨時取消註解來增加對比版本
my_experiments = {
    "V91 (Base)": os.path.join(base_path, "version_91/images"),
    "V92 (Base)": os.path.join(base_path, "version_92/images"),
}

# 指定要展示的 5 個檔案名稱
my_target_files = [
    "b-001372_idx-0.png", 
    "b-001388_idx-0.png", 
    "b-001424_idx-0.png",
    "b-000004_idx-0.png",
    "b-001042_idx-0.png"
]

# 執行腳本
create_scientific_contour_comparison(
    my_experiments, 
    img_size=512, # 論文建議使用 512 以上解析度較清晰
    specified_files=my_target_files,
    output_path=os.path.join(base_path, "thesis_yellow_contour_v4.png")
)