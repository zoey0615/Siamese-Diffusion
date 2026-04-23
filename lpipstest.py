import lpips
import torch
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # 建議加上進度條，可以看到執行進度
import random
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 如果使用多個 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 執行固定種子
seed_everything(42)
# 1. 初始化 LPIPS 模型
# 加上 .eval() 確保模型處於評估模式
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda().eval()

# 2. 影像預處理 (縮放至 256x256 並正規化到 [-1, 1])
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_img(path):
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0).cuda()

# 3. 指定路徑
real_dir = "/media/Siamese-Diffusion/data/train/images_thickv5"
gen_dir = "/media/Siamese-Diffusion/generated_results/version_129/images"

# --- 重點修改：過濾非影像檔案 ---
valid_exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
real_files = sorted([f for f in os.listdir(real_dir) if f.endswith(valid_exts)])
gen_files = sorted([f for f in os.listdir(gen_dir) if f.endswith(valid_exts)])

# 取兩者之中較少的數量，最多取 500 張來平衡準確度與速度
num_samples = min(len(real_files), len(gen_files), 1000)
real_files = real_files[:num_samples]
gen_files = gen_files[:num_samples]

print(f"找到真實影像 {len(real_files)} 張，生成影像 {len(gen_files)} 張。")
print(f"預計比對前 {num_samples} 組樣本...")

distances = []

# 4. 成對計算 (Pairwise)
# 加上 torch.no_grad() 可以節省顯存並加快計算速度
with torch.no_grad():
    for r_file, g_file in tqdm(zip(real_files, gen_files), total=num_samples):
        img0 = load_img(os.path.join(real_dir, r_file))
        img1 = load_img(os.path.join(gen_dir, g_file))
        
        d = loss_fn_vgg(img0, img1)
        distances.append(d.item())

avg_lpips = sum(distances) / len(distances)
print(f"\n========================================")
print(f"Version 12 平均 LPIPS 距離: {avg_lpips:.4f}")
print(f"========================================")