import schedule
import time
import os
import glob
import shutil
import pandas as pd
import sys

# Đảm bảo chạy được từ thư mục gốc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops.retrain import run_retraining
from backend.database import insert_retrain_log

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
UPLOAD_WATCH_DIR = "backend/data"   # Thư mục theo dõi file upload mới
PROCESSED_DIR    = "backend/data/processed"  # Lưu file đã xử lý


# ──────────────────────────────────────────────────────────────────────────────
# JOB 1: Retrain theo lịch 2AM
# ──────────────────────────────────────────────────────────────────────────────
def scheduled_retrain_job():
    print("\n⏰ [2AM SCHEDULER] Bắt đầu tác vụ Retrain định kỳ...")
    run_retraining(trigger="scheduled_2am")


# ──────────────────────────────────────────────────────────────────────────────
# JOB 2: Tự động cập nhật dữ liệu hàng ngày lúc 1AM
# (Kiểm tra có file mới trong backend/data không, nếu có thì merge vào clean_data.csv)
# ──────────────────────────────────────────────────────────────────────────────
def auto_data_update_job():
    """
    Quét thư mục backend/data để tìm file CSV/XLSX mới được upload.
    Nếu có file mới → merge vào clean_data.csv → trigger retrain.
    """
    print("\n📂 [AUTO DATA UPDATE] Đang quét thư mục upload để tìm dữ liệu mới...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Tìm tất cả file CSV/XLSX trong thư mục (không kể thư mục con processed)
    new_files = []
    for ext in ["*.csv", "*.xlsx"]:
        found = glob.glob(os.path.join(UPLOAD_WATCH_DIR, ext))
        new_files.extend(found)

    if not new_files:
        print("ℹ️ [AUTO DATA UPDATE] Không có file mới. Bỏ qua.")
        return

    print(f"📥 [AUTO DATA UPDATE] Tìm thấy {len(new_files)} file mới: {[os.path.basename(f) for f in new_files]}")

    dfs_to_merge = []
    for fpath in new_files:
        try:
            if fpath.endswith(".xlsx"):
                df_new = pd.read_excel(fpath)
            else:
                df_new = pd.read_csv(fpath)
            dfs_to_merge.append(df_new)
            print(f"  ✅ Đọc thành công: {os.path.basename(fpath)} ({len(df_new):,} dòng)")
        except Exception as e:
            print(f"  ⚠️ Lỗi đọc file {os.path.basename(fpath)}: {e}")

    if not dfs_to_merge:
        print("⚠️ [AUTO DATA UPDATE] Không đọc được file nào. Hủy.")
        return

    # Merge với clean_data hiện tại (nếu có)
    if os.path.exists("clean_data.csv"):
        df_existing = pd.read_csv("clean_data.csv")
        all_dfs = [df_existing] + dfs_to_merge
    else:
        all_dfs = dfs_to_merge

    df_merged = pd.concat(all_dfs, ignore_index=True)
    df_merged.drop_duplicates(inplace=True)
    df_merged.to_csv("clean_data.csv", index=False)
    total_rows = len(df_merged)
    print(f"✅ [AUTO DATA UPDATE] Đã merge xong. Tổng số dòng mới: {total_rows:,}")

    # Chuyển file đã xử lý sang thư mục processed
    for fpath in new_files:
        dest = os.path.join(PROCESSED_DIR, os.path.basename(fpath))
        shutil.move(fpath, dest)
        print(f"  📦 Đã chuyển: {os.path.basename(fpath)} → processed/")

    # Trigger retrain ngay sau khi có dữ liệu mới
    insert_retrain_log(
        trigger="data_update",
        status="DATA_MERGED",
        note=f"Merged {len(new_files)} file(s). Total rows: {total_rows:,}"
    )
    print("🔄 [AUTO DATA UPDATE] Kích hoạt Retrain vì có dữ liệu mới...")
    run_retraining(trigger="data_update")


# ──────────────────────────────────────────────────────────────────────────────
# SCHEDULER SETUP
# ──────────────────────────────────────────────────────────────────────────────
def start_scheduler():
    print("="*60)
    print("⏳ MLOps Scheduler đã khởi động.")
    print("   📅 [01:00 AM] Tự động cập nhật dữ liệu mới.")
    print("   📅 [02:00 AM] Retrain mô hình định kỳ.")
    print("   💡 Feedback-triggered retrain: Được gọi trực tiếp từ API.")
    print("="*60)

    # Job 1: Cập nhật dữ liệu tự động lúc 1AM
    schedule.every().day.at("01:00").do(auto_data_update_job)

    # Job 2: Retrain model lúc 2AM (sau khi data đã được update)
    schedule.every().day.at("02:00").do(scheduled_retrain_job)

    # (Để test: schedule.every(2).minutes.do(auto_data_update_job))
    # (Để test: schedule.every(3).minutes.do(scheduled_retrain_job))

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    start_scheduler()
