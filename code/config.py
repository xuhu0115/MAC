# config.py

# 定义人格类型
MBTI_TYPES = {
    "open_personalities": ["ISFP", "INFP", "ENFP", "ESFP", "ENTP"],  # 开放型任务人格
    "complex_personalities": ["INTJ", "INTP", "ISTJ", "ENTJ", "ESTJ"],  # 复杂型任务人格
    "general_round_personalities": ["INFJ", "ISFJ", "ENFJ", "ENTP", "ISTP"]  # 通用人格
}

# 实验配置
EXPERIMENT_CONFIG = {
    "open_task": {
        "rounds": 3,  # 开放性任务的轮次
        "agents_per_round": 2,  # 每轮智能体数量
        "final_round_personalities": MBTI_TYPES["general_round_personalities"]  # 最后一轮使用的智能体人格
    },
    "complex_task": {
        "rounds": 3,  # 复杂性任务的轮次
        "agents_per_round": 2,  # 每轮智能体数量
        "final_round_personalities": MBTI_TYPES["general_round_personalities"]  # 最后一轮使用的智能体人格
    }
}

#定义每个任务类型的默认模型
TASK_MODEL_MAPPING = {
    "open_task": "deepseek",
    "complex_task": "deepseek"
}