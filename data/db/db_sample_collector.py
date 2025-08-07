import pymysql
from config.config import DB_CONFIG

def get_features_from_db():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    try:
        # 用户：保留每个 userId 的最新记录
        cursor.execute("""
            SELECT * FROM user_feature uf
            WHERE uf.id = (
                SELECT MAX(id) FROM user_feature WHERE userId = uf.userId
            )
        """)
        user_records = cursor.fetchall()
        

        # 终端：保留每个 terminalId 的最新记录
        cursor.execute("""
            SELECT * FROM terminal_feature tf
            WHERE tf.id = (
                SELECT MAX(id) FROM terminal_feature WHERE terminalId = tf.terminalId
            )
        """)
        terminal_records = cursor.fetchall()
        

        # 虚拟机：保留每个 resourceId 的最新记录
        cursor.execute("""
            SELECT * FROM vm_feature vf
            WHERE vf.id = (
                SELECT MAX(id) FROM vm_feature WHERE resourceId = vf.resourceId
            )
        """)
        vm_records = cursor.fetchall()
        

        # 所有最近活跃中的连接记录
        cursor.execute("""
            SELECT * FROM connection
            WHERE state = '在线' OR connectEnd IS NULL""")
        connection_records = cursor.fetchall()

        def load_user(user_record):
            return {
                'user_id': user_record['userId'],
                'user_type': user_record['userType'] if user_record['userType'] is not None else 0,
                'login_total': user_record['loginTotal'] if user_record['loginTotal'] is not None else 0,
                'login_succeed': user_record['loginSucceed'] if user_record['loginSucceed'] is not None else 0,
                'if_login_time_ok': user_record['ifLoginTimeOK'] if user_record['ifLoginTimeOK'] not in (None, 0) else 1,
                'login_time_bias': user_record['LoginTimeBias'] if user_record['LoginTimeBias'] is not None else 0,
                'login_time_diff': user_record['LoginTimeDiff'] if user_record['LoginTimeDiff'] is not None else 0,
                'if_ip_allow': user_record['ifIpAllow'] if user_record['ifIpAllow'] not in (None, -2) else 1,
                'if_area_allow': user_record['ifAreaAllow'] if user_record['ifAreaAllow'] is not None else 1
            }

        def load_terminal(terminal_record):
            return {
                'terminal_id': terminal_record['terminalId'],
                'terminal_type': terminal_record['terminalType'] if terminal_record['terminalType'] is not None else 1,
                'user_diff': terminal_record['userDiff'] if terminal_record['userDiff'] is not None else 1
            }
        
        def load_vm(vm_record):
            return {
                "vm_id": vm_record["resourceId"],
                "vm_os_allow": vm_record["VMOsAllow"] if vm_record["VMOsAllow"] is not None else 1,
                "vm_os_version_allow": vm_record["VMOsVersionAllow"] if vm_record["VMOsVersionAllow"] is not None else 1,
                "cpu": vm_record["CPU"] if vm_record["CPU"] is not None else 1,
                "mem": vm_record["mem"] if vm_record["mem"] is not None else 1,
                "vm_connection_user": vm_record["VMConnectionUser"] if vm_record["VMConnectionUser"] is not None else 0,
                "vm_login_total": vm_record["VMLoginTotal"] if vm_record["VMLoginTotal"] is not None else 0,
                "vm_login_succeed": vm_record["VMLoginSucceed"] if vm_record["VMLoginSucceed"] is not None else 0
            }
        
        def load_connection(connection_record):
            return {
                "user_id": connection_record["userId"],
                "terminal_id": connection_record["terminalId"],
                "vm_id": connection_record["resourceId"],
                "connection_id": connection_record["connectionId"],
                "connect_start": str(connection_record["connectStart"]),
                "connect_end": str(connection_record["connectEnd"]) if connection_record["connectEnd"] else "None",
                "online_time": connection_record["onlineTime"],
                "alert_num": 0
            }
        
        sample = {
            "input": {
                "raw_users": [load_user(user_record) for user_record in user_records],
                "raw_terminals": [load_terminal(terminal_record) for terminal_record in terminal_records],
                "raw_vms": [load_vm(vm_record) for vm_record in vm_records],
                "connections": [load_connection(connection_record) for connection_record in connection_records]
            },
            "output": {
                "labels": {
                    user_record["userId"]: "允许访问" for user_record in user_records
                }
            }
        }

    finally:
        cursor.close()
        conn.close()

    return sample
