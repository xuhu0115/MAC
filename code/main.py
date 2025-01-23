# main.py

from task_executor import TaskExecutor
from logger import Logger
from data_loader import DataLoader
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
import os
import json
from config import EXPERIMENT_CONFIG
from evaluation import Evaluator
import re

_ = load_dotenv(find_dotenv())

# def extract_best_answer(data):
#     rounds = EXPERIMENT_CONFIG[task_type]["rounds"]
#     outputs = [output['final_output'] for output in data[rounds-1]]
    
#     print("ouputs[0] type:", type(outputs[0]))
#     print("ouputs[0]:", outputs[0])

#     # 去除多余的 ```json 标记
#     json_str = outputs[0].replace('```json', '').replace('```', '').strip()

#     # 解析 JSON
#     try:
#         data = json.loads(json_str)
#     except json.JSONDecodeError as e:
#         print("JSON 解析错误:", e)
#         print("JSON 字符串内容:", json_str)
#         return []

#     # 提取 Best Answer
#     best_answer = data["Best Answer"]
#     print(best_answer)  # 输出: 翠袖轻摇风似梦

#     best_answers = []
#     return best_answers

if __name__ == "__main__":
    # 初始化日志记录器
    Logger.configure_logger("experiment.log")

    # 实验配置
    task_type = "open_task"  # "open_task" 或 "complex_task"
    specific_task = "poetry"  # 具体任务类型：poetry, story, nli, math
    model_type = "deepseek"
    data_path = "./data/poetry/poetry.json"  # 数据文件路径  "./data/poetry/poetry.json"  "./data/GSM8K/test.jsonl"
    
    # API 密钥
    api_keys = os.environ.get("DEEPSEEK_API_KEY")

    # # Initialize the task executor
    # executor = TaskExecutor(task_type, specific_task, api_keys, model_type, data_path)
    # # Run the task execution process
    # results = executor.run_task()
    # #print("results:",results)
    
    # # # Save the results
    # Logger.save_results(results, f"outputs/{specific_task}_results.json")
    
    # 评估结果
    results_file = "outputs/poetry_results_20250119_202916.json"
    evaluator = Evaluator(task_type, specific_task, api_keys)
    evaluation_results = evaluator.evaluate(results_file)
    
    # 打印评估结果
    print("\nEvaluation Results:")
    for metric, score in evaluation_results.items():
        print(f"{metric}: {score:.4f}")












