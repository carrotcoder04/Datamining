import schedule
import time
import subprocess
import sys
import os

def retrain_job():
    script_path = os.path.join("mlops", "retrain.py")
    print(f"⏰ Đang chạy tác vụ Cronjob: {script_path}")
    subprocess.run([sys.executable, script_path])

def start_scheduler():
    print("⏳ MLOps Scheduler đã khởi động. Lên lịch chạy Retrain mỗi ngày lúc 02:00 AM...")
    
    # Lên lịch chạy mỗi đêm lúc 2h sáng
    schedule.every().day.at("02:00").do(retrain_job)
    
    # (Để test: schedule.every(1).minutes.do(retrain_job))
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    start_scheduler()
