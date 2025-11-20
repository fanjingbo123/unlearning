import subprocess
import time
from tqdm import tqdm
import json
import shutil
import os
def run_command():
    # 要执行的命令
    command = ["python3", "src/exec/unlearn_model.py", "--config-file", "/home/yz979/project/chengye/WAGLE/configs/unlearn/wmdp/GradDiff+WAGLE.json"]

    # 执行命令
    result = subprocess.run(command, capture_output=True, text=True)

    # 打印命令输出
    print(result.stdout)
    print(result.stderr)

def change_config(number):
    file_path = '/home/yz979/project/chengye/WAGLE/configs/unlearn/wmdp/GradDiff+WAGLE.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    data['unlearn']['mask_path'] = '/home/yz979/project/chengye/test/wmdp/chunk_'+str(number)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def repeat_command(times, interval=5):
    for i in tqdm(range(times)):
        change_config(i)
        run_command()
        # 间隔时间
        shutil.rmtree("/home/yz979/scratch/chengye/temp_res/GradDiff")
        time.sleep(interval)

# 设置反复运行次数和间隔时间（单位为秒）
repeat_command(25, interval=10)
