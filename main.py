# main.py

from task_executor import TaskExecutor
from logger import Logger

# main.py

from task_executor import TaskExecutor
from logger import Logger
from data_loader import DataLoader
import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

# if __name__ == "__main__":
#     # 配置
#     task_type = "poetry"  # 可选：poetry, story, nli, math
#     data_file = "data/poetry.json"  # 数据文件路径

#     # 加载数据
#     data = DataLoader.load_data(task_type, data_file)

#     # 初始化 API 密钥
#     api_keys = os.environ.get("DEEPSEEK_API_KEY")


#     # 执行任务
#     executor = TaskExecutor(task_type, api_keys)
#     results = executor.run_task(data)

#     # 保存结果
#     Logger.save_results(results, f"{task_type}_results.json")
    
if __name__ == "__main__":
    # 实验配置
    task_type = "open_task"  # "open_task" 或 "complex_task"
    question = "写一首关于春天的诗："

    # API 密钥
    api_keys = {
        "gpt4": "your-gpt4-api-key",
        "deepseek": "your-deepseek-api-key",
        "qwen2.5": "your-qwen-api-key"
    }

    # 执行任务
    executor = TaskExecutor(task_type, api_keys)
    results = executor.run_task(question)

    # 保存结果
    Logger.save_results(results, f"{task_type}_results.json")