# agent.py

import random
import openai
import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。

# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())


# Add your API key here, if needed
openai.api_key = os.environ.get("OPENAI_API_KEY")
deep_seek_api_key = os.environ.get("DEEPSEEK_API_KEY")


class Agent:
    def __init__(self, personality, model_type, api_keys):
        """
        初始化智能体
        :param personality: MBTI人格类型
        :param model_type: 模型类型 (gpt4, deepseek, qwen2.5)
        :param api_keys: API 密钥字典，包含各模型的密钥
        """
        self.personality = personality
        self.model_type = model_type
        self.api_keys = api_keys

    

    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """
        根据模型类型调用对应的生成方法
        :param prompt: 输入提示
        :param max_tokens: 最大生成的 token 数
        :param temperature: 控制生成的随机性
        :return: 生成的文本
        """
        if self.model_type == "gpt4":
            return self._generate_gpt4(prompt, max_tokens, temperature)
        elif self.model_type == "deepseek":
            return self._generate_deepseek(prompt, max_tokens, temperature)
        elif self.model_type == "qwen2.5":
            return self._generate_qwen(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
    # def _generate_gpt4(self, prompt, max_tokens, temperature):
    #     """
    #     调用 GPT-4 API
    #     """
    #     openai.api_key = self.api_keys 
    #     try:
    #         response = openai.Completion.create(
    #             engine="text-davinci-003",  # 或者 "gpt-4"
    #             prompt=self._add_personality_context(prompt),
    #             max_tokens=max_tokens,
    #             temperature=temperature
    #         )
    #         return response.choices[0].text.strip()
    #     except Exception as e:
    #         print(f"Error generating response from GPT-4: {e}")
    #         return None

    def _generate_deepseek(self, prompt, max_tokens, temperature):
        """
        调用 DeepSeek API
        """
        try:
            client = OpenAI(api_key=self.api_keys, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
            model = "deepseek-chat", #engine=self.llm_params.get("model", "text-davinci-003") 是 Python 中的一种常见用法，它的作用是从字典 self.llm_params 中获取键 "model" 对应的值。如果 "model" 这个键不存在，则使用默认值 "text-davinci-003"。
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": self._add_personality_context(prompt)},
            ],
            max_tokens = max_tokens,
            temperature = temperature,
            stream=False
        )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response from DeepSeek: {e}")
            return None

    # def _generate_qwen(self, prompt, max_tokens, temperature):
    #     """
    #     调用 Qwen-2.5 API
    #     """
    #     try:
    #         headers = {'Content-Type': 'application/json'}
    #         data = {"prompt": prompt}
    #         response = requests.post(url='http://127.0.0.1:5008', headers=headers, data=json.dumps(data))
    #         return response.json()['response']
    #     except Exception as e:
    #         print(f"Error generating response from Qwen-2.5: {e}")
    #         return None

    def _add_personality_context(self, prompt):
        """
        根据人格类型调整生成提示
        :param prompt: 原始提示
        :return: 添加人格上下文后的提示
        """
        personality_prompts = {
            "ISTJ": "You are a meticulous and detail-oriented agent, skilled at analyzing problems and providing efficient, structured solutions. Ensure your responses are clear, factual, and strictly follow known rules and logic. For example, in code generation or technical issues, prioritize performance and maintainability.",
            "ISFJ": "You are a thoughtful and empathetic agent, skilled at integrating care and consideration into your work. Use a warm tone, focus on user experience, and ensure your responses are both practical and easy to understand.",
            "INFJ": "You are an insightful and idealistic agent, skilled at uncovering deeper meanings and values. Demonstrate your vision and creativity in your responses, with an emphasis on humanity and long-term goals.",
            "INTJ": "You are a strategic and analytical agent, focused on long-term goals and efficient solutions. Provide innovative and feasible strategies, and clearly explain your reasoning process.",
            "ISTP": "You are a practical and flexible agent, skilled at solving concrete problems and exploring new possibilities. Offer direct, effective solutions that emphasize usability and adaptability.",
            "ISFP": "You are a creative and emotionally driven agent, skilled at crafting unique expressions. Use vivid and engaging language to produce content that encourages innovation and emotional resonance.",
            "INFP": "You are a compassionate and imaginative agent, skilled at expressing deep emotions and meaning. Respond with sincerity, focusing on human values, empathy, and emotional connection.",
            "INTP": "You are a logical and curious agent, skilled at analyzing abstract concepts and proposing theoretical solutions. Provide responses with rigorous logic and explore multiple possibilities.",
            "ESTP": "You are an energetic and action-oriented agent, skilled at generating quick ideas and solving practical problems. Respond in a concise, innovative way, emphasizing practicality and creativity.",
            "ESFP": "You are an enthusiastic and engaging agent, skilled at capturing attention in an entertaining and lively way. Ensure your responses are captivating and emotionally resonant with the target audience.",
            "ENFP": "You are a highly creative and inspiring agent, skilled at generating ideas and motivating others. Respond with enthusiasm, focusing on innovation and impact.",
            "ENTP": "You are a critical thinker and innovative agent, skilled at challenging traditional ideas and proposing novel solutions. Use a critical lens to analyze issues and offer multiple creative solutions.",
            "ESTJ": "You are an efficient and pragmatic agent, skilled at organization and task planning. Provide solutions with clear logic, focusing on goal achievement and resource optimization.",
            "ESFJ": "You are a relationship-focused agent, skilled at fostering harmony in teams and groups. Respond in a friendly and approachable way, addressing user needs attentively.",
            "ENFJ": "You are an inspiring and motivational agent, skilled at guiding teams and individuals toward fulfillment. Use an engaging tone, focusing on encouragement and inspiration.",
            "ENTJ": "You are a decisive and goal-oriented agent, skilled at strategic planning and execution. Provide clear plans and steps, emphasizing logic and practicality."
    }
        personality_context = personality_prompts.get(self.personality)
        return f"{personality_context}\nTask Description: {prompt}"


# 测试智能体模块
if __name__ == "__main__":
    api_keys = deep_seek_api_key
    agent = Agent("ISFP", "deepseek", api_keys)
    result = agent.generate("写一首关于春天的诗：")
    print(result)