# main.py

from task_executor import TaskExecutor
from logger import Logger

# main.py

from task_executor import TaskExecutor
from logger import Logger
from data_loader import DataLoader
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())
    
if __name__ == "__main__":
    # 实验配置
    task_type = "open_task"  # "open_task" 或 "complex_task"
    model_type = "deepseek"
    data_path = "./data/poetry.json"  # 数据文件路径
    
    # API 密钥
    api_keys = os.environ.get("DEEPSEEK_API_KEY")

    # 执行任务
    executor = TaskExecutor(task_type, api_keys, model_type, data_path)
    results = executor.run_task()

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Logger.save_results(results, f"./outputs/{task_type}_results_{timestamp}.json")