import jieba
import logging
import re
from torch.utils.data import Dataset


# 分词工具类
class WordSegment:
    def __init__(self, prefix='./resources'):
        self.set_jieba(prefix)
        self.stop_words = set()
        self.load_stop_words(prefix + '/stopwords.txt')

        # 设置不可分割的词

    def set_jieba(self, prefix):
        jieba.load_userdict(prefix + "/statename.txt")
        jieba.load_userdict(prefix + "/cityname.txt")
        jieba.load_userdict(prefix + "/distinctname.txt")
        jieba.load_userdict(prefix + "/symbol.txt")
        jieba.load_userdict(prefix + "/namenoise.txt")
        # 自定义词集
        jieba.load_userdict(prefix + '/indiv_words_v2.txt')

    # 外部加载停用词集 file_path=../resources/stopwords.txt
    def load_stop_words(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.stop_words = set(line.strip() for line in file.readlines())

    # 清洗特殊字符
    def clean_text(self, text):
        # 清洗特殊字符
        text = re.sub(r'\(.*?\)|[^a-zA-Z0-9\u4e00-\u9fa5]|(丨)', ' ', str(text))
        # 形如:"EXO店x铺excelAxB" 去除x
        text = re.sub(r'(?<=[\u4e00-\u9fa5])([xX])(?=[\u4e00-\u9fa5])|(?<=[A-Z])x(?=[A-Z])', ' ', text)
        return text

    # 分词
    def cut_word(self, text):
        text = self.clean_text(text)
        # jieba分词
        l_cut_words = jieba.lcut(text)
        # 去除停用词（地名等无用的词）
        out_word_list = [lc_word for lc_word in l_cut_words if
                         lc_word not in self.stop_words and lc_word != '\t' and not lc_word.isspace()]
        # 如果文本去除后，长度变为0，则回滚去除操作
        if out_word_list and (len(out_word_list) != 0):
            return ' '.join(out_word_list)
        else:
            return ' '.join(l_cut_words)


# torch框架下的数据加载器
class DefineDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


# 并行计算时的异常回调
def error_callback(error):
    print(f"Error info: {error}")


class Logger:
    """
    %(levelno)s: 打印日志级别的数值
    %(levelname)s: 打印日志级别名称
    %(pathname)s: 打印当前执行程序的路径，其实就是sys.argv[0]
    %(filename)s: 打印当前执行程序名
    %(funcName)s: 打印日志的当前函数
    %(lineno)d: 打印日志的当前行号
    %(asctime)s: 打印日志的时间
    %(thread)d: 打印线程ID
    %(threadName)s: 打印线程名称
    %(process)d: 打印进程ID
    %(message)s: 打印日志信息
    """

    def __init__(self, logger_name=__name__, log_path='log.log',
                 log_formatter='%(asctime)s - %(levelname)s - %(message)s', log_level=logging.INFO):
        self.logger_name = logger_name  # 日志记录器名字
        self.log_path = log_path  # log文件存储路径
        self.log_formatter = log_formatter  # 日志信息输出格式
        self.log_level = log_level  # 日志级别,级别排序:CRITICAL > ERROR > WARNING > INFO > DEBUG

        self.logger = self.set_logger()
        self.handler = self.set_fileHandler()
        self.logger.addHandler(self.handler)

    # 设置日志记录器
    def set_logger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.log_level)
        return logger

    # 创建文件处理器
    def set_fileHandler(self):
        handler = logging.FileHandler(self.log_path, encoding='utf-8')
        handler.setLevel(self.log_level)
        # 创建日志格式器
        formatter = logging.Formatter(self.log_formatter)
        handler.setFormatter(formatter)
        return handler

    # 添加处理器到记录器
    def set_addHandler(self):
        self.logger.addHandler(self.handler)

    # 输出日志信息
    def info(self, message):
        self.logger.info(message)
