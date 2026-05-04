import subprocess
import sys
import time

def main():
    print("🔥 KHỞI CHẠY HỆ THỐNG MLOPS DATAMINING 🔥")
    
    processes = []
    
    try:
        # 1. Khởi chạy Backend (FastAPI)
        print("🚀 Đang khởi động Backend (FastAPI)...")
        backend_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])
        processes.append(backend_process)
        time.sleep(3) 
        
        # 2. Khởi chạy Frontend (Streamlit)
        print("🚀 Đang khởi động Frontend (Streamlit)...")
        frontend_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.headless", "true"])
        processes.append(frontend_process)
        
        # 3. Khởi chạy MLOps Scheduler
        print("🚀 Đang khởi động MLOps Scheduler...")
        scheduler_process = subprocess.Popen([sys.executable, "mlops/scheduler.py"])
        processes.append(scheduler_process)
        
        print("\n✅ HỆ THỐNG ĐÃ HOẠT ĐỘNG!")
        print("👉 Truy cập Frontend tại: http://localhost:8501")
        print("👉 Truy cập Backend API Docs tại: http://localhost:8000/docs")
        print("\nNhấn Ctrl+C để dừng tất cả dịch vụ.\n")
        
        for p in processes:
            p.wait()
            
    except KeyboardInterrupt:
        print("\n🛑 Đang dừng hệ thống...")
        for p in processes:
            p.terminate()
        print("✅ Đã tắt toàn bộ dịch vụ.")

if __name__ == "__main__":
    main()
