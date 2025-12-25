import time
from apscheduler.schedulers.background import BackgroundScheduler
from data.operations.db_sample_collector import get_features_from_db

scheduler = BackgroundScheduler()
scheduler.add_job(
    get_features_from_db, 
    'interval', 
    seconds=120
)
scheduler.start()

try:
    while True:  # 防止主线程退出
        time.sleep(1)
except KeyboardInterrupt:
    scheduler.shutdown()