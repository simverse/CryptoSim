import sys
import os

# src 디렉토리를 파이썬 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from main import main

if __name__ == "__main__":
    log_file = "backtest_log.txt"
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except OSError as e:
            print(f"Error removing file {log_file}: {e}")
            
    main()
