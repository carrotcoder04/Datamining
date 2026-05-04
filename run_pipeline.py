import os
import subprocess
import sys

def run_script(script_name):
    """Hàm chạy một script python và kiểm tra lỗi"""
    print(f"\n{'='*50}")
    print(f"🚀 ĐANG CHẠY: {script_name}")
    print(f"{'='*50}")
    
    # Sử dụng sys.executable để lấy đường dẫn python hiện tại
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n❌ CÓ LỖI XẢY RA KHI CHẠY: {script_name}. DỪNG PIPELINE.")
        sys.exit(1)
    
    print(f"✅ CHẠY XONG: {script_name}")

def main():
    print("🔥 BẮT ĐẦU CHẠY PIPELINE DATAMINING 🔥\n")
    
    # Bước 1: Xóa các file mô hình cũ để bắt buộc train lại
    checkpoints = ["kmeans_model.pkl", "hierarchical_labels.npy"]
    print("🗑️ Đang kiểm tra và xóa các mô hình checkpoint cũ...")
    for cp in checkpoints:
        if os.path.exists(cp):
            os.remove(cp)
            print(f"  -> Đã xóa: {cp}")
            
    # Bước 2: Danh sách các bước chạy theo thứ tự
    pipeline = [
        # --- ĐỌC DỮ LIỆU TỪ EXCEL ---
        "load_data.py",
        
        # --- TIỀN XỬ LÝ (PREPROCESSING) ---
        "checkNaN.py", 
        "drop_miss_rows.py",
        "drop_negative_quantity.py",
        
        # --- TẠO ĐẶC TRƯNG & CHUẨN HÓA (FEATURE ENGINEERING & SCALING) ---
        "Create_RFM.py",
        "CheckDuplicated.py",
        "remove_outliner.py",
        "scaled_data.py",
        
        # --- GOM CỤM (CLUSTERING) ---
        "kmeans_algorithms.py",
        "hierachical_cluster.py"
    ]
    
    # Bước 3: Vòng lặp chạy từng file
    for script in pipeline:
        if os.path.exists(script):
            run_script(script)
        else:
            print(f"\n⚠️ CẢNH BÁO: Không tìm thấy script {script}. Bỏ qua bước này...")

    print(f"\n{'='*50}")
    print("🎉 PIPELINE HOÀN TẤT THÀNH CÔNG! HÃY KIỂM TRA CÁC FILE KẾT QUẢ VÀ HÌNH ẢNH.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
