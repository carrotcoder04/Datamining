import pandas as pd
import matplotlib.pyplot as plt
import os

def count_rows(filepath, encoding='latin1'):
    try:
        if os.path.exists(filepath):
            # Read only the first column to save memory and speed up
            df = pd.read_csv(filepath, usecols=[0], encoding=encoding)
            return len(df)
        else:
            print(f"File not found: {filepath}")
            return 0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def plot_stages():
    print("Dang dem so luong ban ghi qua tung giai doan... (vui long doi vai giay)")
    
    stages_data = {
        '1. Dữ liệu gốc\n(raw_data.csv)': count_rows('raw_data.csv'),
        '2. Xóa giá trị rỗng\n(drop_miss_row.csv)': count_rows('drop_miss_row.csv'),
        '3. Xóa số lượng âm\n(clean_data.csv)': count_rows('clean_data.csv'),
    }
    
    rfm_stages = {
        '4. Gom nhóm KH (RFM)\n(rfm_data.csv)': count_rows('rfm_data.csv'),
        '5. Xóa ngoại lai\n(final_data.csv)': count_rows('final_data.csv')
    }

    # Bảng 1: Biểu đồ số lượng bản ghi (Dữ liệu giao dịch)
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e']
    bars = plt.bar(stages_data.keys(), stages_data.values(), color=colors)
    
    # Thêm giá trị trên các cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(stages_data.values()) * 0.01), 
                 f'{int(yval):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('Số Lượng Bản Ghi Giao Dịch Qua Các Bước Tiền Xử Lý', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Số lượng bản ghi (dòng)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('bieu_do_1_giao_dich.png', dpi=300)
    print("Da luu: bieu_do_1_giao_dich.png")
    
    # Bảng 2: Biểu đồ số lượng khách hàng (Dữ liệu RFM)
    plt.figure(figsize=(8, 6))
    colors_rfm = ['#2ca02c', '#98df8a']
    bars_rfm = plt.bar(rfm_stages.keys(), rfm_stages.values(), color=colors_rfm, width=0.5)
    
    for bar in bars_rfm:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(rfm_stages.values()) * 0.01), 
                 f'{int(yval):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.title('Số Lượng Khách Hàng (Dữ Liệu RFM)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Số lượng khách hàng', fontsize=12)
    plt.ylim(0, max(rfm_stages.values()) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('bieu_do_2_khach_hang.png', dpi=300)
    print("Da luu: bieu_do_2_khach_hang.png")
    
    plt.show()

if __name__ == "__main__":
    plot_stages()
