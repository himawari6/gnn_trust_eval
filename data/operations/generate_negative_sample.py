import json
import random
from datetime import datetime

# ==== 配置 ====
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
input_file = "data\sample\merged\merged_sample_20250815_101621.json"   # 原始样本文件
output_file = f"data\sample\merged\merged_sample_20250815_101621_modified_at_{current_time}.json"  # 修改后的文件
n_case1 = 50                      # 第一类修改样本数量
n_case2 = 100                      # 第二类修改样本数量
n_case3 = 100                     # 第三类修改样本数量

random.seed(42)


def total_alert_random_distribution(total_alert, num_connections):
    """将总告警数随机分配到多个连接"""
    if num_connections <= 0:
        return []
    parts = [0] * num_connections
    for _ in range(total_alert):
        idx = random.randint(0, num_connections - 1) # 蒙特卡洛，随机落入编号为0~n-1的箱子
        parts[idx] += 1
    return parts


# ==== 1. 读取数据 ====
with open(input_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

print(f"原始样本数: {len(samples)}")

# ==== 2. 找到正常样本（标签全部为允许访问） ====
normal_indices_category_1 = []
for i, sample in enumerate(samples):
    labels = sample["output"]["labels"]
    for uid, lbl in labels.items():
        if lbl != "允许访问":
            break
    else:
        normal_indices_category_1.append(i)

print(f"正常样本数(满足test01/02且允许访问): {len(normal_indices_category_1)}")

# ==== 3. 抽取针对ip_allow设置负面样本的索引 ====
if len(normal_indices_category_1) < (n_case1):
    raise ValueError("针对ip_allow设置的负面样本数量太大，没有足够正面样本")

batch1_idx = random.sample(normal_indices_category_1, n_case1)

# ==== 4. 第一类修改：改 ip 允许 -> 不允许，标签改二次认证 ====
for idx in batch1_idx:
    s = samples[idx]
    for user in s["input"]["raw_users"]:
        user["if_ip_allow"] = -1  # 不允许
    for uid in list(s["output"]["labels"].keys()):
        s["output"]["labels"][uid] = "二次身份认证"

# ==== 5. 找到剩下的有连接之正常样本（标签全部为允许访问） ====
normal_indices_category_2 = []
for i, sample in enumerate(samples):
    if not sample['input']['connections']:
        continue
    labels = sample["output"]["labels"]
    for uid, lbl in labels.items():
        if lbl != "允许访问":
            break
    else:
        normal_indices_category_2.append(i)

selected_indices = random.sample(normal_indices_category_2, n_case2 + n_case3)
batch2_idx = selected_indices[:n_case2]
batch3_idx = selected_indices[n_case2:]


# ==== 5. 第二类修改：connection 总告警数 2-7，随机分配，标签改二次认证 ====
for idx in batch2_idx:
    s = samples[idx]
    total_alert = random.randint(2, 7)
    conn_list = s["input"]["connections"]
    alerts_distribution = total_alert_random_distribution(total_alert, len(conn_list))
    for conn, alerts in zip(conn_list, alerts_distribution):
        conn["alert_num"] = alerts
    s["output"]["labels"]["ff80808197ce5b190197ce5d6d8f0007"] = "二次身份认证"


# ==== 6. 第三类修改：connection 总告警数 8-100，随机分配，标签改拒绝访问 + 随机 CPU/mem 异常 ====
for idx in batch3_idx:
    s = samples[idx]
    total_alert = random.randint(8, 100)
    conn_list = s["input"]["connections"]
    alerts_distribution = total_alert_random_distribution(total_alert, len(conn_list))
    for conn, alerts in zip(conn_list, alerts_distribution):
        conn["alert_num"] = alerts
    s["output"]["labels"]["ff80808197ce5b190197ce5d6d8f0007"] = "限制访问"

    # 随机给 VM 设置异常 CPU/mem
    if random.random() < 0.7:
        for vm in s["input"]["raw_vms"]:
            vm["cpu"] = random.choice([0, -1])
            vm["mem"] = random.choice([0, -1])


# ==== 7. 保存修改后的文件 ====
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(samples, f, ensure_ascii=False, indent=4)

print(f"修改完成，保存到: {output_file}")
