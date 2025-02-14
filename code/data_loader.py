# data_loader.py

import json
import logging
import re

class DataLoader:
    @staticmethod
    def load_data(file_path, specific_task):
        """
        Load data from a file for the specified task type.
        :param file_path: Path to the data file
        :param task_type: Specific task type (e.g., poetry, story, nli, math)
        :return: Processed data
        """
        try:
            # 加载 JSON 或 JSONL 文件
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
            else:
                raise ValueError("Unsupported file format. Only .json and .jsonl are supported.")

            # 根据具体任务类型解析数据
            if specific_task == "poetry":
                return DataLoader._process_poetry_data(data)
            elif specific_task == "story":
                return DataLoader._process_story_data(data)
            elif specific_task == "nli":
                return DataLoader._process_nli_data(data)
            elif specific_task == "math":
                #print("data:",data)
                return DataLoader._process_math_data(data)
            else:
                raise ValueError(f"Unsupported specific task type: {specific_task}")

        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON file: {file_path}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading data: {e}")
            return None

    @staticmethod
    def _process_poetry_data(data):
        """
        处理诗歌生成的数据
        :param data: 数据内容
        :return: 解析后的数据列表
        """
        if "inputs" in data:
            return data["inputs"]
        else:
            raise ValueError("Invalid data format for poetry task. Expected 'inputs' field.")

    @staticmethod
    def _process_story_data(data):
        """
        处理故事创作的数据
        :param data: 数据内容
        :return: 解析后的数据列表
        """
        if "inputs" in data:
            return data["inputs"]
        else:
            raise ValueError("Invalid data format for story task. Expected 'inputs' field.")

    @staticmethod
    def _process_nli_data(data):
        """
        处理自然语言推理的数据
        :param data: 数据内容
        :return: 解析后的数据列表
        """
        if isinstance(data, list) and all("premise" in item and "hypothesis" in item for item in data):
            return data
        else:
            raise ValueError("Invalid data format for NLI task.")

    @staticmethod
    def _process_math_data(data):
        """
        处理数学推理的数据
        :param data: 数据内容
        :return: 解析后的数据列表
        """
        # if isinstance(data, list) and all("question" in item for item in data):
        #     return data
        # else:
        #     raise ValueError("Invalid data format for math task.")
        samples = []
        count = 0
        for i, jsonline in enumerate(data):
            # sample = json.loads(jsonline)
            sample = jsonline
            answer = re.sub(r"[^0-9.]", "",sample["answer"].split("#### ")[1].strip())
            gold_explanation = re.sub('<<.*>>', '', sample["answer"].split("#### ")[0].replace("\n\n", "\n").strip())
            gold_explanation_sents = gold_explanation.split("\n")
            gold_explanation_sents = [gold_explanation_sent + "." if gold_explanation_sent[-1] != "." else gold_explanation_sent for gold_explanation_sent in gold_explanation_sents]
            gold_explanation = " ".join(gold_explanation_sents)
            sample_json = {
                "index": i,
                "question": sample["question"],
                "answer": answer,
                "gold_explanation": gold_explanation
            }
            samples.append(sample_json)
        
        return samples