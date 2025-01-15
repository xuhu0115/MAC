# task_executor.py

from agent import Agent
from data_loader import DataLoader
from config import MBTI_TYPES, EXPERIMENT_CONFIG, TASK_MODEL_MAPPING
from dotenv import load_dotenv, find_dotenv
import random
import os
from typing import List

_ = load_dotenv(find_dotenv())
    
class TaskExecutor:
    def __init__(self, task_type, specific_task, api_keys, model_type, data_path):
        """
        初始化任务执行器
        :param task_type: 任务大类（open_task 或 complex_task）
        :param specific_task: 具体任务类型（poetry, story, nli, math）
        :param api_keys: API 密钥字典
        :param model_type: 模型类型
        :param data_path: 数据文件路径
        """
        self.task_type = task_type
        self.specific_task = specific_task
        self.config = EXPERIMENT_CONFIG[task_type]
        self.api_keys = api_keys
        self.model_type = model_type
        self.data = DataLoader.load_data(data_path, specific_task)  # 根据具体任务类型加载数据

    def run_task(self):
        """
        执行多轮任务
        :return: 每轮的生成结果
        """
        rounds = self.config["rounds"]
        agents_per_round = self.config["agents_per_round"]
        final_round_personalities = self.config["final_round_personalities"]

        # 第 0 轮：初始化智能体
        personalities = MBTI_TYPES["open_personalities"] if self.task_type == ["poetry", "story"] else MBTI_TYPES["complex_personalities"]
        agents, cached_personalities = self._create_agents(personalities, agents_per_round)

        inputs = self._prepare_initial_prompts()  # 初始问题列表 (原始问题, prompt)
        # print("inputs:",inputs)
        results = []

        for round_num in range(rounds):
            print(f"-------------------------Round {round_num + 1}/{rounds}--------------------------")
            # print("inputs:",inputs)
            if round_num == rounds - 1:
                # 最后一轮：使用单个智能体总结并选择最佳答案
                # print("results length:",len(results))
                # print("results:",results)
                # print("results[-1]:",results[-1])
                final_result = self._run_final_round(results[-1], final_round_personalities)
                results.append(final_result)
            else:
                # 第 0 轮到第 rounds-2 轮
                round_results, new_inputs = self._run_round(inputs, agents, cached_personalities)
                results.append(round_results)

                # print("round_results:",round_results)
                # print("results:",results)

                # 修改这里，确保只对 round_results 中的结果进行遍历
                inputs = [
                    self._generate_round_prompt(result)
                    for result in round_results
                    if result["agent_number"] == 0  # 确保每条问题只生成一组输入
                ]

        return results
    
    def _select_personality(self, personalities, agents_per_round):
        """
        随机选择指定数量的人格类型，避免重复选择
        :param personalities: 可选人格类型列表
        :param agents_per_round: 每轮需要的智能体数量
        :return: 随机选择的人格列表
        """
        if len(personalities) >= agents_per_round:
            return random.sample(personalities, agents_per_round)
        else:
            return random.choices(personalities, k=agents_per_round)
    
    def _create_agents(self, personalities, agents_per_round):
        """
        创建智能体列表，并记录对应的人格
        :param personalities: 当前轮次使用的人格类型
        :param agents_per_round: 每轮智能体数量
        :return: 智能体列表和对应的人格
        """
        selected_personalities = self._select_personality(personalities, agents_per_round)
        agents = [Agent(personality, self.model_type, self.api_keys) for personality in selected_personalities]
        return agents, selected_personalities

    def _generate_outputs(self, agents: List[Agent], prompt: str) -> List[str]:
        """
        让智能体生成输出
        :param agents: 智能体列表
        :param prompt: 输入的任务提示
        :return: 输出列表
        """
        return [agent.generate(prompt) for agent in agents]

    def _run_round(self, inputs, agents, cached_personalities):
        """
        执行一轮任务
        :param inputs: 当前轮次的输入
        :param agents: 智能体列表（第 0 轮到第 rounds-2 轮复用）
        :param cached_personalities: 智能体的人格类型列表
        :return: 当前轮次的生成结果
        """
        results = []
        for original_question, prompt in inputs:
            # 每个智能体生成输出
            outputs = self._generate_outputs(agents, prompt)  # 每个智能体生成输出
            # print("original_question type:",type(original_question))
            # print("original_question:",original_question)
            # 记录每个智能体的结果
            question_results = []
            for agent_idx, output in enumerate(outputs):
                other_responses = outputs[:agent_idx] + outputs[agent_idx + 1:]
                if self.specific_task in ["math","nli"]:
                    question_results.append({
                        "question": original_question,  # 原始输入的上半句
                        "input": prompt,  # 本轮的完整 prompt
                        "output": output,  # 当前智能体的输出
                        "other_responses": other_responses,
                        "agent_number": agent_idx, # 当前智能体编号
                        "personalities": cached_personalities[agent_idx]  # 当前智能体的人格类型
                    })
                elif self.specific_task in ["poetry","story"]:
                    question_results.append({
                        "question": original_question,  # 原始输入的上半句
                        "input": prompt,  # 本轮的完整 prompt
                        "output": output,  # 当前智能体的输出
                        "other_responses": other_responses,
                        "agent_number": agent_idx, # 当前智能体编号
                        "personalities": cached_personalities[agent_idx]  # 当前智能体的人格类型
                    })
            # 确保每个问题只生成 agents_per_round 个结果
            #print("question_results:",question_results)
            results.extend(question_results[:len(agents)])
        
        # 为下一轮生成新的 prompts
        new_inputs = [
            self._generate_round_prompt(result) 
            for result in results if result["agent_number"] == 0
        ]

        return results, new_inputs

    def _run_final_round(self, last_round_results, final_round_personalities):
        """
        执行最后一轮任务，对每个问题的多个答案进行总结并决定最优答案
        :param last_round_results: 上一轮的生成结果
        :param final_round_personalities: 最后一轮使用的人格类型
        :return: 每个问题的最终总结结果
        """
        final_results = []
        # print("last_round_results:",last_round_results)
        grouped_results = self._group_by_question(last_round_results)
        
        for question_data in grouped_results:
            
            # 根据任务类型生成总结性提示
            summary_prompt = self._generate_final_prompt(question_data)

            # 使用一个智能体进行总结
            personality = random.choice(final_round_personalities)
            agent = Agent(personality, self.model_type, self.api_keys)
            final_output = agent.generate(summary_prompt)

            # 保存最终输出
            final_results.append({
                "question": question_data["question"],  # 原始输入的上半句
                "input": summary_prompt,  # 完整的 prompt
                "personality": personality,
                "final_output": final_output
            })
        return final_results
    
    def _group_by_question(self, results):
        """
        根据问题分组生成结果
        :param results: 当前轮次的生成结果
        :return: 按问题分组的结果
        """
        # print("results:",results)
        grouped = {}
        for result in results:
            # print("result", result)
            # question = result["question"]
            # print("question:",question)
            # 按问题分组
            # if question_key not in grouped:
            #     grouped[question_key] = {"question": result["question"], "responses": []}

            if self.specific_task in ["math","nli"]:
                question_key = result["question"]["question"]
                if question_key not in grouped:
                    grouped[question_key] = {"question": result["question"]["question"], "responses": []}
            elif self.specific_task in ["poetry","story"]:
                question_key = result["question"]
                if question_key not in grouped:
                    grouped[question_key] = {"question": result["question"], "responses": []}
            
            grouped[question_key]["responses"].append(result["output"])
        return list(grouped.values())

    def _prepare_initial_prompts(self):
        """
        根据任务类型生成初始 prompt
        :return: 初始 prompt 列表
        """
        #print("self.data:",self.data)
        self.data = self.data[:1]

        prompts = []
        for data_item in self.data:
            if self.specific_task == "poetry":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            elif self.specific_task == "story":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            elif self.specific_task == "nli":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            elif self.specific_task == "math":
                prompts.append((
                    data_item,
                    self._generate_prompt(data_item)
                ))
            else:
                raise ValueError(f"Unsupported specific task type: {self.specific_task}")
        
        return prompts

    def _generate_round_prompt(self, result):
        """
        根据任务类型生成中间轮次的 prompt
        :param result: 上一轮的结果
        :param task_type: 当前任务类型
        :return: 新一轮的 prompt
        """
        if self.task_type == "open_task":
            # 开放性任务：诗歌生成或故事创作
            if self.specific_task == "poetry":  # 诗歌生成
                return (result["question"],  # 原始问题
                        "你的任务是创作优美的中国古诗词。以下是你之前的回答以及其他智能体的回答，请你结合它们的内容，重新思考并创作出更加精妙的下一句。"
                        "\n\n**要求如下：**\n"
                        "1. **综合参考**：结合你自己的回答和其他智能体的回答，从中汲取优点，避免重复或简单拼接，做到在内容上融合创造。\n"
                        "2. **保持原创性与连贯性**：新的句子必须是原创的，同时与上一句诗意境连贯，语言典雅，吻合古诗词的风格。\n"
                        "3. **提升质量**：力求语言更精炼、意境更深远、韵律更和谐，体现更高水平的文采。\n"
                        "4. **严格输出格式**：仅输出单独的一句诗，不添加其他内容。\n\n"
                        f"**你的回答：** {result['output']}\n"
                        f"**其他智能体的回答：** {', '.join(result['other_responses'])}\n\n"
                        "请你结合以上内容，重新创作出更高质量的下一句：")
            elif self.specific_task == "story":  # 故事创作
                return (result["question"],  # 原始问题
                        "你的任务是根据关键词创作一个精彩的故事。以下是你之前的回答以及其他智能体的回答，请你综合这些内容，优化和提升原故事，使其更加吸引人。"
                        "\n\n**要求如下：**\n"
                        "1. **综合参考**：结合你自己的回答和其他智能体的回答，从中汲取优点，避免重复或简单拼接，做到在内容上融合创造。\n"
                        "2. **提升吸引力**：优化故事情节，使其更加生动、有趣，吸引读者注意力。\n"
                        "3. **语言生动**：用词需更加生动有趣，符合故事创作的文风。\n"
                        "4. **严格输出格式**：仅输出完整的故事内容，不添加其他说明。\n\n"
                        f"**你的回答：** {result['output']}\n"
                        f"**其他智能体的回答：** {', '.join(result['other_responses'])}\n\n"
                        "请你结合以上内容，优化并重写完整的故事：")

        elif self.task_type == "complex_task":
            # 复杂性任务：自然语言推理或数学推理
            if self.specific_task == "nli":  # 自然语言推理
                return (result["question"],  # 原始问题
                        "根据以下的自然语言推理任务，请对多个智能体的回答进行综合分析，并给出最终的结论。"
                        "\n\n**要求如下：**\n"
                        "1. **综合分析**：参考你自己的回答和其他智能体的回答，分析它们的优缺点。\n"
                        "2. **准确性**：确保最终结论准确无误，符合语义逻辑。\n"
                        "3. **严格输出格式**：仅输出“蕴含”、“矛盾”或“中立”，不添加其他内容。\n\n"
                        f"**前提：** {result['question']}\n"
                        f"**假设：** {result['input']}\n"
                        f"**你的回答：** {result['output']}\n"
                        f"**其他智能体的回答：** {', '.join(result['other_responses'])}\n\n"
                        "请你结合以上内容，给出最终的结论：")
            elif self.specific_task == "math":  # 数学推理
                return (result["question"],  # 原始问题
                        "以下是针对一个数学问题的多个解答，请你综合分析，并给出最终的全面解答。"
                        "\n\n**要求如下：**\n"
                        "1. **综合分析**：参考你自己的回答和其他智能体的回答，分析它们是否正确。\n"
                        "2. **完整性**：确保最终解答步骤完整、逻辑清晰，答案准确无误。\n"
                        "3. **严格输出格式**：仅输出完整的解答步骤，最后以“#### 最终答案”格式标注答案。\n\n"
                        f"**问题：** {result['question']}\n"
                        f"**你的回答：** {result['output']}\n"
                        f"**其他智能体的回答：** {', '.join(result['other_responses'])}\n\n"
                        "请你结合以上内容，重写完整的解答步骤：")

        raise ValueError(f"Unsupported task type: {self.task_type}")

    def _generate_prompt(self, data_item):
        """
        根据任务类型生成 prompt
        :param task_type: 任务类型
        :param data_item: 当前数据项
        :param round_num: 当前轮次
        :return: prompt 文本
        """
        # print("data_item:",data_item)
        if self.task_type == "open_task":
            if self.specific_task == "poetry":  # 诗歌生成
                return (
                    "你是一位才华横溢的古代诗人，擅长创作优美的中国古诗词。"
                        "现在，我将提供一句古诗作为开头，请你根据其意境、韵律和风格，创作出下一句，使其成为一首和谐美丽的诗句。"
                        "\n\n请注意以下要求：\n"
                        "1. **保持原创性**：创作的句子必须完全原创，而不是引用已有的古诗词。\n"
                        "2. **符合韵律**：保持与上一句一致的平仄规则和押韵要求。\n"
                        "3. **延续意境**：下一句应与上一句的意境连贯，体现自然、情感或哲理之美。\n"
                        "4. **语言优美**：用词需典雅，符合古诗词的文风和艺术感。\n"
                        "5. **严格输出格式**：仅输出单独的一句诗，不添加其他内容。\n\n"
                        f"**上一句：** {data_item}\n**下一句：**"
                )
            elif self.specific_task == "story":  # 故事生成
                return (
                    "你是一位富有想象力的作家，擅长创作引人入胜的故事。现在，我将提供一些关键词，请你根据这些关键词写一个短篇故事。\n\n"
                    "**要求：**\n"
                    "1. 故事必须包含所有关键词，并合理融入情节中。\n"
                    "2. 故事需有清晰的起因、经过和结局，情节连贯且富有逻辑。\n"
                    "3. 语言生动，注重细节描写，吸引读者。\n\n"
                    f"**关键词**：{', '.join(data_item)}\n**请开始你的创作：**"
                )

        elif self.task_type == "complex_task":
            if self.specific_task == "nli":  # 自然语言推理
                return (
                    "你是一位擅长逻辑推理的助手，能够分析前提和假设之间的关系。现在，我将提供一对句子，请你判断它们的关系，并说明理由。\n\n"
                    "**要求：**\n"
                    "1. 判断关系是“蕴含”、“矛盾”还是“中立”。\n"
                    "2. 输出判断结果，并附上逻辑分析，解释原因。\n\n"
                    f"**前提**：{data_item['premise']}\n"
                    f"**假设**：{data_item['hypothesis']}\n\n"
                    "**请开始分析：**"
                )
            elif self.specific_task == "math":  # 数学推理
                # question = data_item["question"]
                return (
                    "你是一位擅长数学的助理，能够清晰地解释复杂的数学问题。现在，我将提供一个数学问题，请你按照以下要求进行解答。\n\n"
                    "**要求：**\n"
                    "1. 写出详细的解题步骤，每一步都清晰且有逻辑。\n"
                    "2. 最后输出答案，格式为“答案：XXX”。\n"
                    "3. 如果问题涉及单位，答案需包含单位。\n\n"
                    f"**问题**：{data_item['question']}\n\n"
                    "**请开始解答：**"
                )

        raise ValueError(f"Unsupported task type: {self.specific_task}")

    def _generate_final_prompt(self, question_data):
        """
        根据任务类型生成最后一轮的总结性 prompt
        :param question_data: 某个问题分组的结果
        :param task_type: 当前任务类型
        :return: 最后一轮总结性 prompt
        """
        if self.task_type == "open_task":
            if self.specific_task == "poetry":  # 诗歌生成
                return (f"上半句：{question_data['question']}\n"
                        "以下是多个智能体针对上半句创作的诗句，请你总结分析并选择一个最优答案：\n"
                        + "\n".join([f"智能体{idx+1}的回答：{response}" for idx, response in enumerate(question_data["responses"])])
                        + "\n\n请根据以下要求选择最佳答案并解释理由："
                        "\n1. **意境深远**"
                        "\n2. **语言优美**"
                        "\n3. **韵律和谐**"
                        "\n4. **创新与连贯性**"
                        "\n\n请严格按照以下格式输出："
                        "{最终答案: , 解释: }")
            elif self.specific_task == "story":  # 故事创作
                return (f"关键词：{question_data['question']}\n"
                        "以下是多个智能体针对关键词创作的故事，请总结分析并选择一个最优答案：\n"
                        + "\n".join([f"智能体{idx+1}的回答：{response}" for idx, response in enumerate(question_data["responses"])])
                        + "\n\n请根据以下要求选择最佳答案并解释理由："
                        "\n1. **情节完整**"
                        "\n2. **吸引力**"
                        "\n3. **语言生动**"
                        "\n\n请严格按照以下格式输出："
                        "{最终答案: , 解释: }")

        elif self.task_type == "complex_task":
            if self.specific_task == "nli":  # 自然语言推理
                return (f"前提：{question_data['question']}\n"
                        "以下是多个智能体针对假设的推理结果，请总结分析并选择一个最优答案：\n"
                        + "\n".join([f"智能体{idx+1}的回答：{response}" for idx, response in enumerate(question_data["responses"])])
                        + "\n\n请根据以下要求选择最佳答案并解释理由："
                        "\n1. **准确性**"
                        "\n2. **逻辑性**"
                        "\n\n请严格按照以下格式输出："
                        "{最终答案: , 解释: }")
            elif self.specific_task == "math":  # 数学推理
                return (f"问题：{question_data['question']}\n"
                        "以下是多个智能体针对问题的解答，请总结分析并选择一个最优答案：\n"
                        + "\n".join([f"智能体{idx+1}的回答：{response}" for idx, response in enumerate(question_data["responses"])])
                        + "\n\n请根据以下要求选择最佳答案并解释理由："
                        "\n1. **准确性**"
                        "\n2. **解答完整性**"
                        "\n\n请严格按照以下格式输出："
                        "{最终答案: , 解释: }")

        raise ValueError(f"Unsupported task type: {self.specific_task}")


# 测试任务执行模块
# if __name__ == "__main__":
#     api_keys = os.environ.get("DEEPSEEK_API_KEY")
#     executor = TaskExecutor("open_task", api_keys)
#     question = "上半句：白日依山尽，\n请你根据自己的理解创作下一句，不要直接抄袭原文内容：{下半句}(只需要给出下一句内容，不需要分析)"
#     results = executor.run_task(question)
    