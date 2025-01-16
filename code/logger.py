# logger.py

import logging
import json
from datetime import datetime

class Logger:
    @staticmethod
    def configure_logger(log_file="experiment.log"):
        """
        配置日志记录器
        :param log_file: 日志文件名
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        logging.info("Logger initialized.")

    @staticmethod
    def save_results(results, filename="outputs/results.json"):
        """
        Save experiment results to a JSON file.
        :param results: Results to save
        :param filename: File path for saving the results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename.replace(".json", f"_{timestamp}.json")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logging.info(f"Results saved to {filename}")
        except Exception as e:
            logging.error(f"Failed to save results: {e}")