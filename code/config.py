# config.py

# Define personality types
MBTI_TYPES = {
    "open_personalities": ["ISFP", "INFP", "ENFP", "ESFP", "ENTP"],  # Personalities for open tasks
    "complex_personalities": ["INTJ", "INTP", "ISTJ", "ENTJ", "ESTJ"],  # Personalities for complex tasks
    "general_round_personalities": ["INFJ", "ISFJ", "ENFJ", "ENTP", "ISTP"]  # Generic personalities for final rounds
}

# Experiment configuration
EXPERIMENT_CONFIG = {
    "open_task": {
        "rounds": 3,  # Total number of rounds
        "agents_per_round": 2,  # Number of agents per round
        "final_round_personalities": MBTI_TYPES["general_round_personalities"]  # Personalities for final round
    },
    "complex_task": {
        "rounds": 5,  
        "agents_per_round": 6,  
        "final_round_personalities": MBTI_TYPES["general_round_personalities"]  
    }
}

# Mapping of task types to model types
TASK_MODEL_MAPPING = {
    "open_task": "deepseek",
    "complex_task": "deepseek"
}