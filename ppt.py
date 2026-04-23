import os
import cv2
import numpy as np

def create_version_comparison(version_name, img_dir, mask_dir, output_dir, target_filenames):
    """
    為單一版本建立對比圖：
    第一列：生成的影像
    第二列：純遮罩 (Mask)
    第三列：影像 + 淡黃色輪廓疊加 (Overlay)
    """
    images = []
    masks = []
    overlays = []
    
    # 設定輪廓顏色 (BGR 格式：淡黃色)
    contour_color = (150, 255, 255) 

    for fname in target_filenames:
        img_path = os.path.join(img_dir, fname)
        mask_fname = fname.rsplit('.', 1)[0] + '.png' 
        mask_path = os.path.join(mask_dir, mask_fname)
        
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 以灰階讀取遮罩
        
        if img is not None and mask is not None:
            # 1. 統一縮放
            img = cv2.resize(img, (512, 512))
            mask = cv2.resize(mask, (512, 512))
            
            # 2. 製作淡黃色輪廓 (利用 OpenCV 膨脹相減法)
            # 二值化處理
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            # 模擬 PIL MaxFilter: 使用核心大小決定線條位置與粗細
            kernel_outer = np.ones((9, 9), np.uint8)
            kernel_inner = np.ones((5, 5), np.uint8)
            
            outer_edge = cv2.dilate(binary_mask, kernel_outer, iterations=1)
            inner_edge = cv2.dilate(binary_mask, kernel_inner, iterations=1)
            
            # 相減得到邊緣線條
            edge_mask = cv2.subtract(outer_edge, inner_edge)
            
            # 3. 建立疊加圖 (Overlay)
            overlay_img = img.copy()
            # 在有邊緣的地方填入淡黃色
            overlay_img[edge_mask > 0] = contour_color
            
            # 4. 準備拼接影像
            # 遮罩轉回 3 通道以便拼接
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            images.append(img)
            masks.append(mask_3ch)
            overlays.append(overlay_img)
        else:
            print(f"⚠️ 警告：在版本 {version_name} 中找不到檔案 {fname}，已跳過。")
            
    if not images:
        return

    # 橫向拼接
    row_images = np.hstack(images)
    row_masks = np.hstack(masks)
    row_overlays = np.hstack(overlays)
    
    # 縱向拼接 (原圖 -> 遮罩 -> 輪廓疊加)
    final_comparison = np.vstack([row_images, row_masks, row_overlays])
    
    # 加入版本文字標記 (選配)
    cv2.putText(final_comparison, f"Version: {version_name}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Comparison_{version_name}.jpg")
    cv2.imwrite(save_path, final_comparison)
    print(f"✅ 已儲存 {version_name} 對比圖 (包含輪廓疊加) 至: {save_path}")

def main():
    # ================= 配置區域 =================
    selected_filenames = [
        "b-001115_idx-0.png",
        "b-001111_idx-0.png",
        "b-000883_idx-0.png",
        "b-000915_idx-0.png",
        "b-000922_idx-0.png"
    ]

    base_results = "/media/Siamese-Diffusion/generated_results"
    
    # 版本字典：方便維護路徑
    versions = {
        "V12": "version_12",   #
        "V130": "version_130", #
        "V129": "version_129", #
        "V148": "version_148",
        "V146": "version_146",
        "V145": "version_145",
        "V150": "version_150",
        "V151": "version_151",
        "V152": "version_152",
    }
    
    output_root = "./experiment_results_overlay"
    # ===========================================

    print(f"🚀 開始產生疊加對比圖...")

    for v_name, v_folder in versions.items():
        img_dir = os.path.join(base_results, v_folder, "images")
        if v_name in ["V12", "V130", "V129"]:
            mask_folder_name = "masks"
        else:
            mask_folder_name = "original_copied_masks"
        mask_dir = os.path.join(base_results, v_folder, mask_folder_name)
        
        # 為了除錯，建議印出路徑看看
        print(f"檢查版本 {v_name}: Mask 目錄為 {mask_dir}")
        
        create_version_comparison(v_name, img_dir, mask_dir, output_root, selected_filenames)

if __name__ == "__main__":
    main()