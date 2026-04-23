import json
from tqdm import tqdm

# ================= 配置設定 =================
INPUT_JSON = '/media/Siamese-Diffusion/data/train_cvcv5.json'
OUTPUT_JSON = '/media/Siamese-Diffusion/data/train_cvcv5_clean.json'
# ===========================================

def clean_json_by_prompt():
    cleaned_entries = []
    
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="檢查 Prompt 中"):
        entry = json.loads(line.strip())
        prompt = entry.get('prompt', '')
        
        # --- 核心邏輯 ---
        # 1. 確保字串裡有 "CVC"
        # 2. 確保字串裡沒有其他干擾管線的關鍵字 (妳可以根據需求增減清單)
        bad_keywords = ["Swan Ganz", "NGT", "ETT", "Gastric", "Tracheal"]
        
        has_cvc = "CVC" in prompt
        has_junk = any(word in prompt for word in bad_keywords)
        
        if has_cvc and not has_junk:
            cleaned_entries.append(entry)
            
    # 儲存結果
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        for entry in cleaned_entries:
            f.write(json.dumps(entry) + '\n')
            
    print(f"✨ 清洗完成！原本: {len(lines)} 筆 -> 剩下: {len(cleaned_entries)} 筆")
    print(f"🗑️ 已剔除 {len(lines) - len(cleaned_entries)} 筆 Prompt 含有雜質的資料。")

if __name__ == "__main__":
    clean_json_by_prompt()