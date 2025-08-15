import pymysql
import json
from datetime import datetime, timedelta
from config.config import DB_CONFIG

USER_BATCH_SIZE = 3
TERM_BATCH_SIZE = 2
VM_BATCH_SIZE = 2

# 特征表更新时间间隔（分钟）
FEATURE_UPDATE_INTERVAL = 2

def get_latest_update_id(cursor, table):
    """获取特征表最新一批的起始id"""
    cursor.execute(f"SELECT id FROM {table} ORDER BY id DESC LIMIT 1")
    latest_id = cursor.fetchone()["id"]
    return latest_id

def fetch_batch_by_offset(cursor, table, batch_size, offset_batches):
    """从最新批次往回 offset_batches 个批次取数据"""
    offset = offset_batches * batch_size
    cursor.execute(f"""
        SELECT * FROM {table} ORDER BY id DESC LIMIT {batch_size} OFFSET {offset}
    """)
    return cursor.fetchall()

def fetch_connections_for_time(cursor, t):
    """取 t 前 10 分钟的连接"""
    window_start = t - timedelta(minutes=10)
    cursor.execute("""
        SELECT * FROM connection
        WHERE connectStart BETWEEN %s AND %s
    """, (window_start, t))
    return cursor.fetchall()

def make_sample(user_records, terminal_records, vm_records, connection_records):
    """拼装样本"""
    def load_user(u):
        return {
            'user_id': u['userId'],
            'user_type': u['userType'] if u['userType'] is not None else 0,
            'login_total': u['loginTotal'] or 0,
            'login_succeed': u['loginSucceed'] or 0,
            'if_login_time_ok': u['ifLoginTimeOK'] if u['ifLoginTimeOK'] not in (None, 0) else 1,
            'login_time_bias': u['LoginTimeBias'] or 0,
            'login_time_diff': u['LoginTimeDiff'] or 0,
            'if_ip_allow': u['ifIpAllow'] if u['ifIpAllow'] not in (None, -2) else 1,
            'if_area_allow': u['ifAreaAllow'] if u['ifAreaAllow'] is not None else 1
        }

    def load_terminal(t):
        return {
            'terminal_id': t['terminalId'],
            'terminal_type': t['terminalType'] if t['terminalType'] is not None else 1,
            'user_diff': t['userDiff'] if t['userDiff'] is not None else 1
        }

    def load_vm(v):
        return {
            "vm_id": v["resourceId"],
            "vm_os_allow": v["VMOsAllow"] if v["VMOsAllow"] is not None else 1,
            "vm_os_version_allow": v["VMOsVersionAllow"] if v["VMOsVersionAllow"] is not None else 1,
            "cpu": v["CPU"] if v["CPU"] is not None else 1,
            "mem": v["mem"] if v["mem"] is not None else 1,
            "vm_connection_user": v["VMConnectionUser"] or 0,
            "vm_login_total": v["VMLoginTotal"] or 0,
            "vm_login_succeed": v["VMLoginSucceed"] or 0
        }

    def load_connection(c):
        return {
            "user_id": c["userId"],
            "terminal_id": c["terminalId"],
            "vm_id": c["resourceId"],
            "connection_id": c["connectionId"],
            "connect_start": str(c["connectStart"]),
            "connect_end": str(c["connectEnd"]) if c["connectEnd"] else "None",
            "online_time": c["onlineTime"],
            "alert_num": 0
        }

    return {
        "input": {
            "raw_users": [load_user(r) for r in user_records],
            "raw_terminals": [load_terminal(r) for r in terminal_records],
            "raw_vms": [load_vm(r) for r in vm_records],
            "connections": [load_connection(r) for r in connection_records]
        },
        "output": {
            "labels": {r["userId"]: "允许访问" for r in user_records}
        }
    }

def backfill_samples(start_time_str, end_time_str, last_update_time_str, save_dir="data/sample"):
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
    last_update_time = datetime.strptime(last_update_time_str, "%Y-%m-%d %H:%M:%S")

    # 获取最新批次的ID
    latest_user_id = get_latest_update_id(cursor, "user_feature")
    latest_term_id = get_latest_update_id(cursor, "terminal_feature")
    latest_vm_id = get_latest_update_id(cursor, "vm_feature")

    samples = []
    t = start_time
    while t <= end_time:
        # 找到 t 对应的特征更新时间
        minutes_diff = int((last_update_time - (t - timedelta(seconds=100))).total_seconds() // 60)  # 偏移101秒保证落到正确批次
        offset_batches = minutes_diff // FEATURE_UPDATE_INTERVAL

        # 按 offset_batches 从特征表取数据
        user_records = fetch_batch_by_offset(cursor, "user_feature", USER_BATCH_SIZE, offset_batches)
        terminal_records = fetch_batch_by_offset(cursor, "terminal_feature", TERM_BATCH_SIZE, offset_batches)
        vm_records = fetch_batch_by_offset(cursor, "vm_feature", VM_BATCH_SIZE, offset_batches)

        # 取连接数据
        connection_records = fetch_connections_for_time(cursor, t)

        # 生成样本
        sample = make_sample(user_records, terminal_records, vm_records, connection_records)
        samples.append(sample)

        # 保存单个文件
        file_time = t.strftime("%Y%m%d_%H%M%S")
        with open(f"{save_dir}/sample_{file_time}.json", "w", encoding="utf-8") as f:
            json.dump([sample], f, ensure_ascii=False, indent=4)

        t += timedelta(minutes=2)

    cursor.close()
    conn.close()

    print(f"补采完成，共 {len(samples)} 个样本")
    return samples

if __name__ == "__main__":
    # 从 19:24:24 到 21:20:24 补采，最后一次特征更新时间是 22:20:43
    backfill_samples(
        "2025-08-14 13:53:11",
        "2025-08-14 14:11:11",
        "2025-08-14 14:19:31"
    )