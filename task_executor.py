# task_executor.py

from agent import Agent
from data_loader import DataLoader
from config import MBTI_TYPES, EXPERIMENT_CONFIG, TASK_MODEL_MAPPING
from dotenv import load_dotenv, find_dotenv
import random
import os

_ = load_dotenv(find_dotenv())
    
class TaskExecutor:
    def __init__(self, task_type, api_keys):
        """
        初始化任务执行器
        :param task_type: 任务类型（open_task 或 complex_task）
        :param api_keys: API 密钥字典
        """
        self.task_type = task_type
        self.config = EXPERIMENT_CONFIG[task_type]
        self.api_keys = api_keys

    def run_task(self, question):
        """
        执行多轮任务
        :param question: 输入的初始问题
        :return: 每轮的生成结果
        """
        rounds = self.config["rounds"]
        agents_per_round = self.config["agents_per_round"]
        final_round_personalities = self.config["final_round_personalities"]

        # 初始输入
        inputs = [question]

        results = []
        for round_num in range(rounds):
            print(f"-------------------------Round {round_num + 1}/{rounds}--------------------------")

            # 确定本轮使用的人格
            if round_num == rounds - 1:
                personalities = final_round_personalities
                print("Using general-round personalities for final round.")
            else:
                personalities = MBTI_TYPES["open_personalities"] if self.task_type == "open_task" else MBTI_TYPES["complex_personalities"]

            # 每轮生成
            round_results = self._run_round(inputs, personalities, agents_per_round)
            print("round_results:", round_results)
            results.append(round_results)

            # 更新下一轮的输入：每个智能体根据上一轮自己的输出和其他智能体的输出生成新的内容
            inputs = [f"你的回答：{result['output']}\n其他智能体的回答：{', '.join(result['other_responses'])}\n请你结合其他智能体的回答，再次重新思考，给出回答。" for result in round_results]
        
        return results
    def _create_agents(self, personalities, agents_per_round, model_type):
        """
        创建智能体列表
        :param personalities: 当前轮次使用的人格类型
        :param agents_per_round: 每轮智能体数量
        :param model_type: 模型类型 ("gpt4", "deepseek", "qwen2.5")
        :return: 智能体列表
        """
        return [Agent(random.choice(personalities), model_type, self.api_keys) for _ in range(agents_per_round)]

    def _generate_outputs(self, agents, prompt):
        """
        让智能体生成输出
        :param agents: 智能体列表
        :param prompt: 输入的任务提示
        :return: 输出列表
        """
        return [agent.generate(prompt) for agent in agents]

    def _run_round(self, inputs, personalities, agents_per_round):
        results = []
        for prompt in inputs:
            model_type = TASK_MODEL_MAPPING.get(self.task_type, "gpt4")
            agents = self._create_agents(personalities, agents_per_round, model_type)
            outputs = self._generate_outputs(agents, prompt)

            # 每个智能体的输出和其他智能体的输出
            for idx, output in enumerate(outputs):
                other_responses = outputs[:idx] + outputs[idx+1:]
                results.append({
                    "input": prompt,
                    "output": output,
                    "other_responses": other_responses
                })
        return results

# 测试任务执行模块
# if __name__ == "__main__":
#     api_keys = os.environ.get("DEEPSEEK_API_KEY")
#     executor = TaskExecutor("open_task", api_keys)
#     question = "上半句：白日依山尽，\n请你根据自己的理解创作下一句，不要直接抄袭原文内容：{下半句}(只需要给出下一句内容，不需要分析)"
#     results = executor.run_task(question)
    