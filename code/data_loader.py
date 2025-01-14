# data_loader.py

import json
import logging

class DataLoader:
    @staticmethod
    def load_data(file_path):
        """
        加载 JSON 数据，并处理文件不存在或格式错误的情况
        :param file_path: 数据文件路径
        :return: 数据内容
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".json"):
                    return json.load(f)
                elif file_path.endswith(".jsonl"):
                    return [json.loads(line) for line in f]
                else:
                    raise ValueError("Unsupported file format. Only .json and .jsonl are supported.")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON file: {file_path}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading data: {e}")
            return None