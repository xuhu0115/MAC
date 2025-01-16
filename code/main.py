# main.py

from task_executor_ch_p import TaskExecutor
from logger import Logger
from data_loader import DataLoader
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os

_ = load_dotenv(find_dotenv())
    
if __name__ == "__main__":
    # 初始化日志记录器
    Logger.configure_logger("experiment.log")

    # 实验配置
    task_type = "complex_task"  # "open_task" 或 "complex_task"
    specific_task = "math"  # 具体任务类型：poetry, story, nli, math
    model_type = "deepseek"
    data_path = "./data/GSM8K/test.jsonl"  # 数据文件路径  "./data/poetry.json"  "./data/GSM8K/test.jsonl"
    
    # API 密钥
    api_keys = os.environ.get("DEEPSEEK_API_KEY")

    # Initialize the task executor
    executor = TaskExecutor(task_type, specific_task, api_keys, model_type, data_path)
    # Run the task execution process
    results = executor.run_task()

    # Save the results
    Logger.save_results(results)






