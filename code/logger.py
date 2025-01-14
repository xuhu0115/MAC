# logger.py

import json
from datetime import datetime

class Logger:
    @staticmethod
    def save_results(results, filename="results.json"):
        """
        保存结果到 JSON 文件
        :param results: 实验结果
        :param filename: 文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename.replace(".json", f"_{timestamp}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {filename}")