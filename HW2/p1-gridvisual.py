"""
from chatgpt 
"""

from PIL import Image

# 定義照片的資料夾路徑和照片數量
folder_path = "./H2O/p1-output/svhn"  # 更換成您的資料夾路徑
num_rows = 10
num_cols = 10
output_width = 280  # 輸出照片的寬度

# 創建一個空白的輸出圖片
output_image = Image.new('RGB', (output_width, output_width))

for row in range(num_rows):
    for col in range(num_cols):
        
        # 計算當前照片的文件名
        image_name = f"{row}_{col+1:03d}.png"
        image_path = f"{folder_path}/{image_name}"

        try:
            # 打開當前照片
            image = Image.open(image_path)
        except IOError:
            print(f"無法打開照片 {image_name}")
            continue

        # 計算每張照片在輸出圖片中的位置
        x = col * (output_width // num_cols)
        y = row * (output_width // num_rows)

        # 將當前照片貼到輸出圖片的對應位置
        output_image.paste(image, (x, y))

# 儲存輸出圖片
output_image.save("gridvisual-2.png")
