# main.py


import re
import pandas as pd
import random
import openai
import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from agent import Agent
from task_executor import TaskExecutor
from logger import Logger
from data_loader import DataLoader
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from config import EXPERIMENT_CONFIG
from typing import List, Dict, Any, Union
from pathlib import Path
import pypinyin
import logging
import numpy as np

# Load environment variables from local/project configuration.
_ = load_dotenv(find_dotenv())

# Add your API key here, if needed
openai.api_key = os.environ.get("OPENAI_API_KEY")
deep_seek_api_key = os.environ.get("DEEPSEEK_API_KEY")

class LLM:
    def __init__(self, model_type, api_keys):
        """
        Initialize the agent.
        :param personality: MBTI personality type
        :param model_type: Model type (e.g., gpt4, deepseek, qwen2.5)
        :param api_keys: Dictionary containing API keys for the models
        """
        
        self.model_type = model_type
        self.api_keys = api_keys

    def generate(self, prompt, max_tokens=1024, temperature=0.7):
        """
        Generate a response based on the model type.
        :param prompt: Input prompt
        :param max_tokens: Maximum tokens for the response
        :param temperature: Controls randomness of the output
        :return: Generated text
        """
        if self.model_type == "gpt4":
            return self._generate_gpt4(prompt, max_tokens, temperature)
        elif self.model_type == "deepseek":
            return self._generate_deepseek(prompt, max_tokens, temperature)
        elif self.model_type == "qwen2.5":
            return self._generate_qwen(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    def _generate_gpt4(self, prompt, max_tokens, temperature):
        """
        Call the Deepseek API.
        """
        openai.api_key = self.api_keys 
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",  # or "Deepseek"
                prompt=self._add_personality_context(prompt),
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating response from Deepseek: {e}")
            return None

    # def _generate_deepseek(self, prompt, max_tokens, temperature):
    #     """
    #     Call the DeepSeek API.
    #     """
    #     try:
    #         client = OpenAI(api_key=self.api_keys, base_url="https://api.deepseek.com")
    #         response = client.chat.completions.create(
    #         model = "deepseek-chat", #engine=self.llm_params.get("model", "text-davinci-003") 是 Python 中的一种常见用法，它的作用是从字典 self.llm_params 中获取键 "model" 对应的值。如果 "model" 这个键不存在，则使用默认值 "text-davinci-003"。
    #         messages=[
    #             {"role": "system", "content": "You are a helpful assistant"},
    #             {"role": "user", "content": prompt},
    #         ],
    #         max_tokens = max_tokens,
    #         temperature = temperature,
    #         stream=False
    #     )
    #         return response.choices[0].message.content
    #     except Exception as e:
    #         print(f"Error generating response from DeepSeek: {e}")
    #         return None

    def _generate_deepseek(self, prompt, max_tokens, temperature):
        """
        Call the DeepSeek API.
        max_tokens: default:512, Required range: 1 < x < 8192
        temperature: default: 0.7
        top_k: default:50
        """
        try:
            client = OpenAI(api_key="sk-gwompcsazrqpbcbwhsbwzwojdtzjuftaehxefbglzpawfmmg", base_url="https://api.siliconflow.cn/v1")
            response = client.chat.completions.create(
                model='deepseek-ai/DeepSeek-V3',
                messages=[
                    {'role': 'user', 
                    'content': prompt}
                ],
                max_tokens = max_tokens,
                temperature = temperature,
                stream=False
            )
            # for chunk in response:
            #     print(chunk.choices[0].delta.content, end='')
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response from DeepSeek: {e}")
            return None
        

    def _generate_qwen(self, prompt, max_tokens, temperature):
        """
        Call the Qwen-2.5 API.
        """
        try:
            headers = {'Content-Type': 'application/json'}
            data = {"prompt": prompt}
            response = requests.post(url='http://127.0.0.1:5008', headers=headers, data=json.dumps(data))
            return response.json()['response']
        except Exception as e:
            print(f"Error generating response from Qwen-2.5: {e}")
            return None


def _generate_prompt(data_item, task_type, specific_task):
    """
    Generate prompt according to task type
    :param data_item: Current data item
    :param round_num: Current round
    :return: prompt 
    """
    # print("data_item:",data_item)
    if task_type == "open_task":
        if specific_task == "poetry":  # Poetry Generation
            return (
f"""
You are a highly talented Chinese ancient poet, renowned for composing exquisite Chinese poetry. I will provide the opening line of a poem, and your task is to craft the next line based on its artistic conception, rhythm, and style, creating a harmonious and beautiful continuation.
    
Please adhere to the following requirements:
1. **Originality**: Your line must be entirely original, not borrowed from any existing poetry.
2. **Rhythmic Consistency**: Ensure the tonal pattern and rhyme scheme align perfectly with the previous line.
3. **Artistic Continuity**: Your line should seamlessly extend the meaning, atmosphere, and imagery of the given line, evoking a sense of beauty in nature, emotion, or philosophy.
4. **Elegance in Language**: Employ refined and elegant diction that reflects the style and artistry of classical Chinese poetry.
5. **Strict Output Format**: Only output a single poetic line, without any additional explanations or context.

**Opening Line:** {data_item}
**Next Line:**
"""
            )
        elif specific_task == "story":  # Story Generation
            return (
                "You are an imaginative and skilled storyteller, renowned for crafting captivating and engaging narratives. I will provide you with a set of keywords, and your task is to write a short story that seamlessly incorporates these keywords into its plot."
                "\n\nPlease adhere to the following requirements:\n"
                "1. **Incorporate All Keywords**: The story must include all the provided keywords, integrating them naturally and meaningfully into the storyline.\n"
                "2. **Structured Narrative**: Ensure the story has a clear beginning, middle, and end, with a logical and coherent progression.\n"
                "3. **Engaging and Vivid Language**: Use descriptive and vivid language to bring the story to life and captivate the audience.\n"
                "4. **Creativity**: Demonstrate originality and creativity, making the story unique and memorable.\n\n"
                f"**Keywords**: {', '.join(data_item)}\n**Start your story:**"
            )

    elif task_type == "complex_task":
        if specific_task == "nli":  # NLI
            return (
                "You are a logical reasoning assistant with exceptional skills in analyzing the relationship between a premise and a hypothesis. I will provide you with a pair of sentences, and your task is to determine their relationship and explain your reasoning."
                "\n\nPlease adhere to the following requirements:\n"
                "1. **Relationship Determination**: Identify whether the relationship is 'entailment', 'neutral', or 'contradiction', and map it to the corresponding label value:\n"
                "   - 'entailment' → label value: 0\n"
                "   - 'neutral' → label value: 1\n"
                "   - 'contradiction' → label value: 2\n"
                "2. **Output Format**: Strictly output only the label value and a concise explanation of your reasoning.\n"
                "   Example:\n"
                "   - 0: The hypothesis logically follows from the premise.\n"
                "   - 1: The hypothesis is unrelated or lacks sufficient information to infer from the premise.\n"
                "   - 2: The hypothesis contradicts the premise.\n\n"
                f"**Premise**: {data_item['premise']}\n"
                f"**Hypothesis**: {data_item['hypothesis']}\n\n"
                "**Start your analysis and provide your output:**"
            )
        elif specific_task == "math":  # Mathematical Reasoning
            # question = data_item["question"]
            return (
                "You are an expert mathematics assistant, skilled at clearly explaining and solving complex mathematical problems. I will provide you with a problem, and your task is to solve it step by step while adhering to the following requirements."
                "\n\nPlease adhere to the following requirements:\n"
                "1. **Detailed Steps**: Provide a step-by-step solution, ensuring each step is logical, detailed, and easy to follow.\n"
                "2. **Final Answer Format**: Clearly state the final answer at the end in the format 'Answer: XXX'.\n"
                "3. **Units**: If the problem involves units, include them in your final answer.\n"
                "4. **Clarity and Precision**: Make sure your explanation is concise, accurate, and free of ambiguity.\n\n"
                f"**Problem**: {data_item['question']}\n\n"
                "**Start solving:**"
            )

    raise ValueError(f"Unsupported task type: {specific_task}")
    
def _prepare_initial_prompts(data, task_type, specific_task):
    """
    Generate initial prompt according to task type
    :return: Initial prompt list
    """
    #print("self.data:",self.data)
    #data = data[:3]

    prompts = []
    for data_item in data:
        if specific_task == "poetry":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
            # print("prompts:",prompts)
        elif specific_task == "story":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
        elif specific_task == "nli":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
        elif specific_task == "math":
            prompts.append((
                data_item,
                _generate_prompt(data_item, task_type, specific_task)
            ))
        else:
            raise ValueError(f"Unsupported specific task type: {specific_task}")
    
    return prompts

def _run_round(inputs, agent, max_tokens=1024, temperature=0.7):
    """
    Execute one round of the task.
    :param inputs: Inputs for the current round
    :param agents: List of agents (reused from round 0 to rounds-2)
    :param cached_personalities: List of agent personalities
    :return: Results of the current round
    """
    question_results = []
    for original_question, prompt in inputs:
        # print("prompt:", prompt)
        # print("original_question:", original_question)
        # Each agent generates outputs
        output = agent.generate(prompt, max_tokens, temperature)
        #print("output:",output)

        question_results.append({
            "question": original_question,  # The first half of the original input
            "input": prompt,  # The complete prompt for this round
            "output": output,  # Current agent output
        })

    return question_results

class Evaluator:
    def __init__(self, task_type: str, specific_task: str, api_key: str, model_type: str):
        """
        初始化评测器
        :param task_type: 任务类型 (open_task/complex_task)
        :param specific_task: 具体任务 (poetry/story/nli/math)
        :param api_key: Deepseek API密钥
        """
        self.task_type = task_type
        self.specific_task = specific_task
        self.api_key = api_key
        self.model_type = model_type
        self.llm = LLM(self.model_type, self.api_key)
        
        # 创建结果保存目录
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志记录
        logging.basicConfig(
            filename=self.results_dir / f"{specific_task}_exp1_no-mbti-mac_evaluation.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def evaluate(self, results):
        """
        评估任务结果
        :param results_file: 任务结果文件路径
        :return: 评估结果字典
        """
        
        #print("results:",results)
        if self.task_type == "open_task":
            if self.specific_task == "poetry":
                return self._evaluate_poetry(results)
            elif self.specific_task == "story":
                return self._evaluate_story(results)
        else:  # complex_task
            if self.specific_task == "nli":
                return self._evaluate_nli(results)
            elif self.specific_task == "math":
                return self._evaluate_math(results)
                
        raise ValueError(f"Unknown task type: {self.task_type} or specific task: {self.specific_task}")
    
    def parse_json_from_markdown(self, s):
        # 匹配 ```json 和 ``` 之间的内容（支持跨行）
        match = re.search(r'```json(.*?)```', s, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError("Invalid JSON format") from e
        raise ValueError("No JSON code block found")


    def _evaluate_poetry(self, results: List[Dict]) -> Dict[str, float]:
        """评估诗歌生成结果"""
        total_scores = []
        #inputs, outputs = self.extract_inputs_outputs(results)
        # print("inputs:",inputs)
        # print("outputs:",outputs)

        for result in results:
            # 保存原始输入和输出用于人工验证
            #self._save_for_human_validation(result)
            
            scores = {
                "coherence": self._evaluate_coherence_llm(result['question'], result['output']),
                "rhythm": self._evaluate_rhythm(result['output']),
                "elegance": self._evaluate_elegance_llm(result['question'], result['output']),
                "emotion": self._evaluate_emotion_llm(result['question'], result['output'])
            }
            total_scores.append(scores)
            
        # # 计算平均分数
        # avg_scores = {
        #     metric: np.mean([score[metric] for score in total_scores])
        #     for metric in ["coherence", "rhythm", "elegance", "emotion"]
        # }
        # avg_scores["overall"] = np.mean(list(avg_scores.values()))
        
        return total_scores
    
    def _evaluate_story(self, results: List[Dict]) -> Dict[str, float]:
        """评估故事生成结果"""
        total_scores = []
        
        for result in results:
            #self._save_for_human_validation(result)
            
            scores = {
                "relevance": self._evaluate_relevance_llm(result["input"], result["final_output"]),
                "interestingness": self._evaluate_interestingness_llm(result["final_output"]),
                "fluency": self._evaluate_fluency_llm(result["final_output"]),
                "coherence": self._evaluate_coherence_llm(result["input"], result["final_output"]),
                "completeness": self._evaluate_completeness_llm(result["final_output"])
            }
            total_scores.append(scores)
            
        avg_scores = {
            metric: np.mean([score[metric] for score in total_scores])
            for metric in ["relevance", "interestingness", "fluency", "coherence", "completeness"]
        }
        avg_scores["overall"] = np.mean(list(avg_scores.values()))
        
        return avg_scores

    def _evaluate_nli(self, results: List[Dict]) -> Dict[str, float]:
        """评估NLI任务结果"""
        correct = 0
        total = len(results)
        
        for result in results:
            self._save_for_human_validation(result)
            
            # 验证预测标签是否正确
            if result["final_output"] == result["label"]:
                correct += 1
                
        accuracy = correct / total
        return {"accuracy": accuracy}

    def _evaluate_math(self, results: List[Dict]) -> Dict[str, float]:
        """评估数学推理任务结果"""
        correct = 0
        total = len(results)
        
        for result in results:
            self._save_for_human_validation(result)
            
            # 提取数字答案并比较
            predicted = self._extract_number(result["final_output"])
            reference = self._extract_number(result["answer"])
            
            if predicted is not None and reference is not None:
                # 允许一定的数值误差
                if abs(predicted - reference) < 1e-6:
                    correct += 1
                    
        accuracy = correct / total
        return {"accuracy": accuracy}
    def multi_response(self, prompt, max_attempts=5, max_tokens=1024, temperature=0.7):
        attempt = 0

        while attempt < max_attempts:
            try:
                response = self.llm.generate(prompt, max_tokens, temperature)
                return self.parse_json_from_markdown(response)
            except Exception as e:
                attempt += 1
                if attempt == max_attempts:
                    logging.error(f"Deepseek evaluation error after {max_attempts} attempts: {e}")
                    return 9999  # 所有尝试失败后返回默认分数
                else:
                    logging.error(f"Attempt {attempt} failed, retrying...: {e}")
                    # 可以选择在这里添加一个短暂的等待时间，例如使用time.sleep(1)，避免过快地进行重试
                

    def _evaluate_coherence_llm(self, input_text: str, output_text: str):
        """使用Deepseek评估连贯性"""
        prompt = f"""Dear Evaluator,

        Please professionally assess the coherence and logical consistency of the given text, providing a score between 0.0 and 1.0 (where 1.0 indicates complete coherence, and 0.0 indicates a complete lack thereof), to two decimal places:

        Text Part 1: {input_text}
        Text Part 2: {output_text}

        Evaluation Criteria:
        - Whether the text structure is clear and reasonable;
        - Whether there is a strong logical connection between Text Part 1 and Text Part 2;
        - Whether the content remains consistent throughout, without contradictions.

        Please strictly adhere to the following JSON output format:
            {{
            "score": "<Your Score Here>",
            "Explanation": "<Reasons for the score>"
            }}
        Important: Ensure the JSON output is valid, compact, and easy to parse. Do not include extra line breaks, indentation, or additional commentary outside the JSON.

        Thank you for your professional judgment and cooperation.
        """
        
        response = self.multi_response(prompt, max_attempts=5, max_tokens=1024, temperature=0.7)
        return response

    def _evaluate_rhythm(self, text: str) -> float:
        """评估诗歌韵律"""
        # 使用pypinyin获取拼音
        try:
            pinyins = pypinyin.pinyin(text, style=pypinyin.TONE)
            # 检查韵脚
            # 这里是简化版的实现，实际应用中可能需要更复杂的韵律规则
            return 1.0  # 返回韵律评分
        except Exception as e:
            logging.error(f"Rhythm evaluation error: {e}")
            return 9999

    def _extract_number(self, text: str) -> Union[float, None]:
        """从文本中提取数值答案"""
        try:
            # 寻找数字答案的模式
            patterns = [
                r'####\s*(\d+\.?\d*)',  # 匹配 #### 后的数字
                r'答案[:：]\s*(\d+\.?\d*)',  # 匹配"答案："后的数字
                r'(\d+\.?\d*)$'  # 匹配末尾的数字
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return float(match.group(1))
            return None
        except Exception as e:
            logging.error(f"Number extraction error: {e}")
            return None

    def _save_for_human_validation(self, result: Dict):
        """保存结果供人工验证"""
        filename = self.results_dir / f"{self.specific_task}_human_validation.jsonl"
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')

    # 其他Deepseek评估方法实现类似
    def _evaluate_elegance_llm(self, input_text: str, output_text: str):
        """评估语言优美性"""
        prompt = f"""Dear Evaluator,

        Please assess the aesthetic quality of the language in the given text below, providing a score between 0.0 and 1.0 (where 1.0 represents the highest level of linguistic beauty, and 0.0 indicates none):

        Text: {input_text}+','+{output_text}+'。'

        Evaluation Criteria:
        - The elegance of vocabulary used;
        - The gracefulness of expression;
        - The depth of the imagery conveyed.

        Please strictly adhere to the following JSON output format:\n
            {{
            "score": "<Your Score Here>",
            "Explanation": "<Reasons for the score>"
            }}
        Important: Ensure the JSON output is valid, compact, and easy to parse. Do not include extra line breaks, indentation, or additional commentary outside the JSON.


        Instructions: Ensure the score is provided to two decimal places and nothing else follows it.

        Thank you for your professional judgment and cooperation."""

        response = self.multi_response(prompt, max_attempts=5, max_tokens=1024, temperature=0.7)
        return response

    def _evaluate_emotion_llm(self, input_text: str, output_text: str):
        """评估情感主题一致性"""
        prompt = f"""Dear Evaluator,

        Please evaluate the consistency of sentiment and theme between the input and output texts below, providing a score between 0.00 and 1.00 (where 1.00 indicates perfect consistency, and 0.00 indicates no consistency):

        Input Text: {input_text}
        Output Text: {output_text}

        Evaluation Criteria:
        - Whether the expression of sentiment is unified;
        - Whether the themes are consistent;
        - Whether the imagery conveyed is harmonious.

        Please strictly adhere to the following JSON output format:\n
            {{
            "score": "<Your Score Here>",
            "Explanation": "<Reasons for the score>"
            }}
        Important: Ensure the JSON output is valid, compact, and easy to parse. Do not include extra line breaks, indentation, or additional commentary outside the JSON.

        Thank you for your professional judgment and cooperation."""

        response = self.multi_response(prompt, max_attempts=5, max_tokens=1024, temperature=0.7)
        return response
    
    # def clean_json_string(self, json_string):
    #     # Remove the leading "```json" and any trailing whitespace
    #     json_str = json_string.replace("```json", "").strip()
    #     # Parse the JSON data
    #     data = json.loads(json_str)
        
    #     return cleaned


    def extract_inputs_outputs(self, results):
        
        # 提取 inputs 和 outputs
        inputs = [result['question'] for result in results]
        outputs = [result['output'] for result in results]

        return inputs, outputs
    
            
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

    # ------------------------------------------------------
    data = DataLoader.load_data(data_path, specific_task)

    agent = LLM(model_type = model_type, api_keys = api_keys)
    inputs = _prepare_initial_prompts(data, task_type, specific_task)
    #print("inputs:", inputs)
    question_results = _run_round(inputs, agent, max_tokens=1024, temperature=1.5)  #poetry/story:1.5; NLI/math:0.0

    # print("question_results:", question_results)
    # 定义CSV文件名
    filename = 'outputs/exp1_no-mbti-mac.csv'
    # 将数据转换为 DataFrame
    df = pd.DataFrame(question_results)
    # 写入 CSV 文件
    df.to_csv(filename, index=False)
    print(f"Data has been written to {filename}")

    # ------------------------------------------------------
    # question_results = pd.read_csv(filename)
    # # 将 DataFrame 转换为字典列表，恢复 question_results 变量
    # question_results = question_results.to_dict(orient='records')
    # #print("question_results:", question_results)
    # evaluator = Evaluator(task_type, specific_task, api_keys, model_type)
    # evaluation_results = evaluator.evaluate(question_results)

    # df = pd.DataFrame(evaluation_results)
    # # 写入 CSV 文件
    # df.to_csv('outputs/exp1_eva_no-mbti-mac.csv', index=False)
    # print(f"Data has been written to outputs/exp1_eva_no-mbti-mac.csv")
    
    
    



