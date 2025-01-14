# task_executor.py

from agent import Agent
from data_loader import DataLoader
from config import MBTI_TYPES, EXPERIMENT_CONFIG, TASK_MODEL_MAPPING
from dotenv import load_dotenv, find_dotenv
import random
import os

_ = load_dotenv(find_dotenv())
    
class TaskExecutor:
    def __init__(self, task_type, api_keys, model_type, data_path):
        """
        初始化任务执行器
        :param task_type: 任务类型（open_task 或 complex_task）
        :param api_keys: API 密钥字典
        :param data_path: 数据文件路径
        """
        self.task_type = task_type
        self.config = EXPERIMENT_CONFIG[task_type]
        self.api_keys = api_keys
        self.model_type = model_type
        self.data = DataLoader.load_data(data_path)

    def run_task(self):
        """
        执行多轮任务
        :return: 每轮的生成结果
        """
        rounds = self.config["rounds"]
        agents_per_round = self.config["agents_per_round"]
        final_round_personality = self.config["final_round_personalities"]

        inputs = self._prepare_initial_prompts()
        results = []

        for round_num in range(rounds):
            print(f"-------------------------Round {round_num + 1}/{rounds}--------------------------")

            # 确定本轮使用的人格
            if round_num == rounds - 1:
                # 最后一轮：使用单个智能体总结并选择最佳答案
                final_result = self._run_final_round(results[-1], final_round_personality)
                results.append(final_result)
            else:
                personalities = MBTI_TYPES["open_personalities"] if self.task_type == "open_task" else MBTI_TYPES["complex_personalities"]
                # 每轮生成
                round_results = self._run_round(inputs, personalities, agents_per_round)
                results.append(round_results)

                # 更新下一轮的输入
                inputs = [
                    (
                        "你的任务是创作优美的中国古诗词。以下是你之前的回答以及其他智能体的回答，请你结合它们的内容，重新思考并创作出更加精妙的下一句。"
                        "\n\n**要求如下：**\n"
                        "1. **综合参考**：结合你自己的回答和其他智能体的回答，从中汲取优点，避免重复或简单拼接，做到在内容上融合创造。\n"
                        "2. **保持原创性与连贯性**：新的句子必须是原创的，同时与上一句诗意境连贯，语言典雅，吻合古诗词的风格。\n"
                        "3. **提升质量**：力求语言更精炼、意境更深远、韵律更和谐，体现更高水平的文采。\n"
                        "4. **严格输出格式**：仅输出单独的一句诗，不添加其他内容（如解释或注释）。\n\n"
                        f"**你的回答：** {result['output']}\n"
                        f"**其他智能体的回答：** {', '.join(result['other_responses'])}\n\n"
                        "请你结合以上内容，重新创作出更高质量的下一句："
                    )
                    for result in round_results
                ]

        return results
    
    def _create_agents(self, personalities, agents_per_round):
        """
        创建智能体列表
        :param personalities: 当前轮次使用的人格类型
        :param agents_per_round: 每轮智能体数量
        :param model_type: 模型类型 ("gpt4", "deepseek", "qwen2.5")
        :return: 智能体列表
        """
        #return [Agent(random.choice(personalities), self.model_type, self.api_keys) for _ in range(agents_per_round)]
        agents = []
        personalities_agents = []
        for _ in range(agents_per_round):
            personality = random.choice(personalities)
            agent = Agent(personality, self.model_type, self.api_keys)
            agents.append(agent)
            personalities_agents.append(personality)

        return agents,personalities_agents

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
            model_type = TASK_MODEL_MAPPING.get(self.task_type)
            agents, personalities_agents = self._create_agents(personalities, agents_per_round)
            outputs = self._generate_outputs(agents, prompt)

            # 每个智能体的输出和其他智能体的输出
            for idx, output in enumerate(outputs):
                other_responses = outputs[:idx] + outputs[idx+1:]
                results.append({
                    "question":self.data["inputs"][idx],
                    "input": prompt,
                    "output": output,
                    "other_responses": other_responses,
                    "agent_number":idx,
                    "personalities":personalities_agents[idx]
                })
        return results

    def _run_final_round(self, last_round_results, personalities):
        """
        执行最后一轮任务，对每个问题的多个答案进行总结并决定最优答案
        :param last_round_results: 上一轮的生成结果
        :param personalities: 最后一轮使用的人格类型
        :return: 每个问题的最终总结结果
        """
        final_results = []
        questions = self._group_by_question(last_round_results)
        print("questions:",questions)

        for question_idx, question_data in enumerate(questions):
            # 准备总结的提示，包含该问题的所有上一轮答案
            summary_prompt = f"问题{question_idx + 1}：{question_data['input']}\n以下是多个智能体对该问题的回答，请总结并决定一个最优答案，并解释原因。\n"
            for idx, response in enumerate(question_data["responses"]):
                summary_prompt += f"智能体{idx + 1}的回答：{response}\n"
            summary_prompt += "请输出最终答案，并严格按照以下格式：{最终答案:, 解释:}"

            # 只使用一个智能体进行总结
            personality = random.choice(personalities)
            agent = Agent(personality, self.model_type, self.api_keys)
            final_output = agent.generate(summary_prompt)

            # 保存最终输出
            final_results.append({
                "question":self.data["inputs"][question_idx],
                "input": question_data["input"],
                "personality":personality,
                "final_output": final_output
            })

        return final_results
    
    def _group_by_question(self, last_round_results):
        """
        根据问题分组上一轮的结果
        :param last_round_results: 上一轮的生成结果
        :return: 分组后的问题及其对应的回答
        """
        grouped_results = {}
        for result in last_round_results:
            input_text = result["input"]
            if input_text not in grouped_results:
                grouped_results[input_text] = {"input": input_text, "responses": []}
            grouped_results[input_text]["responses"].append(result["output"])
        return list(grouped_results.values())
        
    def _prepare_initial_prompts(self):
        """
        根据任务类型生成初始 prompt
        :return: 初始 prompt 列表
        """
        if self.task_type == "open_task":
            # 处理诗歌生成
            if "inputs" in self.data and isinstance(self.data["inputs"][0], str):
                return [
                    (
                        "你是一位才华横溢的古代诗人，擅长创作优美的中国古诗词。"
                        "现在，我将提供一句古诗作为开头，请你根据其意境、韵律和风格，创作出下一句，使其成为一首和谐美丽的诗句。"
                        "\n\n请注意以下要求：\n"
                        "1. **保持原创性**：创作的句子必须完全原创，而不是引用已有的古诗词。\n"
                        "2. **符合韵律**：保持与上一句一致的平仄规则和押韵要求。\n"
                        "3. **延续意境**：下一句应与上一句的意境连贯，体现自然、情感或哲理之美。\n"
                        "4. **语言优美**：用词需典雅，符合古诗词的文风和艺术感。\n"
                        "5. **严格输出格式**：仅输出单独的一句诗，不添加其他内容。\n\n"
                        f"**上一句：** {line}\n**下一句：**"
                    )
                    for line in self.data["inputs"]
                ]
            # 处理故事创作
            elif "inputs" in self.data and isinstance(self.data["inputs"][0], list):
                return [f"关键词：{', '.join(keywords)}\n请根据这些关键词写一个故事：" for keywords in self.data["inputs"]]

        elif self.task_type == "complex_task":
            # 处理自然语言推理
            if isinstance(self.data, list) and "premise" in self.data[0] and "hypothesis" in self.data[0]:
                return [f"前提：{item['premise']}\n假设：{item['hypothesis']}\n请判断其关系是“蕴含”、“矛盾”还是“中立”：" for item in self.data]
            # 处理数学推理
            elif isinstance(self.data, list) and "question" in self.data[0]:
                return [f"问题：{item['question']}\n请写出详细的解答步骤：" for item in self.data]

        # 如果未匹配到任务类型或数据格式，抛出异常
        raise ValueError(f"Unsupported task type or data format. Task type: {self.task_type}, Data: {self.data}")
    

# 测试任务执行模块
# if __name__ == "__main__":
#     api_keys = os.environ.get("DEEPSEEK_API_KEY")
#     executor = TaskExecutor("open_task", api_keys)
#     question = "上半句：白日依山尽，\n请你根据自己的理解创作下一句，不要直接抄袭原文内容：{下半句}(只需要给出下一句内容，不需要分析)"
#     results = executor.run_task(question)
    