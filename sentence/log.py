import sys
import logging
class LoggerWriter:
    def __init__(self, level, model_name_or_path, poison):
        # 创建logger
        model = next((k for k in ("vicuna", "mistral", "opt", "Qwen2", "llama") if k in model_name_or_path), None)
        # 根据模型名称和poison状态构造日志文件名
        filename = f"./log/{model}_{poison}.log"
        self.logger = logging.getLogger("mylogger")
        self.logger.setLevel(logging.DEBUG)
        # 创建写入日志文件的handler
        fh = logging.FileHandler(filename, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        # 创建输出到控制台的handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # 日志级别
        self.level = level

    def write(self, message):
        # 处理print函数调用的输出
        if message.rstrip() != "":
            self.logger.log(self.level, message.rstrip())

    def flush(self):
        # 满足文件对象接口
        pass

    def fileno(self):
        # 返回一个有效的文件描述符
        return sys.stdout.fileno()

# 创建一个LoggerWriter实例

