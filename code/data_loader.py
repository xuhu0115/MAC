# data_loader.py

import json

class DataLoader:
    @staticmethod
    def load_data(file_path):
        """
        加载 JSON 数据
        :param file_path: 数据文件路径
        :return: 数据内容
        """
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".json"):
                return json.load(f)
            elif file_path.endswith(".jsonl"):
                return [json.loads(line) for line in f]
            else:
                raise ValueError("Unsupported file format. Only .json and .jsonl are supported.")