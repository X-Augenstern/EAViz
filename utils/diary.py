from logging import Formatter, DEBUG, INFO, WARNING, ERROR, CRITICAL, getLogger, StreamHandler, FileHandler
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from sys import stdout
from os.path import join, exists, getctime
from os import mkdir, listdir, remove
from datetime import datetime

# 时间 - 日志器名 - 文件名_行号 - 调用函数名 - 日志级别 - 文本内容
LOG_FORMAT = Formatter('%(asctime)s - %(name)s - %(filename)s_[line:%(lineno)d] - %(funcName)s - %(levelname)s'
                       '\n%(message)s\n')
LOG_PREFIX = 'diary_'
LOG_ENCODING = 'utf-8'


class Diary:
    def __init__(self, name=f"{LOG_PREFIX}{datetime.now().strftime('%Y%m%d%H%M%S')}", stream_level='WARNING',
                 save=True, save_level='DEBUG', log_folder_path='.\\diary', update_freq='D', save_interval=1,
                 backup_count=7):
        """
        初始化自定义Log类
        :param name: logger名称，会在console与.log文件中输出
        :param stream_level: console只会输出不低于此等级的log信息
        :param save: 是否利用文件句柄进行处理
        :param save_level: 进行存储的日志级别
        :param log_folder_path: 日志存储的文件夹路径
        :param update_freq: 日志更新频率
        :param save_interval: 日志存储间隔
        :param backup_count: 允许备份文件数量上限
        """

        def get_logging_level(level_str):
            if level_str == 'DEBUG':
                return DEBUG
            elif level_str == 'INFO':
                return INFO
            elif level_str == 'WARNING':
                return WARNING
            elif level_str == 'ERROR':
                return ERROR
            else:
                return CRITICAL

        self.name = name
        self.stream_level = get_logging_level(stream_level)
        self.save = save
        self.save_level = get_logging_level(save_level)
        self.log_folder_path = log_folder_path
        self.update_freq = update_freq
        self.save_interval = save_interval
        self.backup_count = backup_count

    def init_logger(self, fh_type='1'):
        """
        创建日志器
        :param fh_type: 文件句柄类型
        """
        logger = getLogger(self.name)
        # 设定logger默认级别
        logger.setLevel(DEBUG)
        # 判断句柄是否已经被创建，如果当前句柄为空，则可以添加
        # if not logger.handlers:
        logger.addHandler(self.init_streamHd())

        if self.save:
            log_file_path = join(self.log_folder_path, self.name + '.log')
            # 是否存在此路径文件夹，不存在—>创建
            if not exists(self.log_folder_path):
                mkdir(self.log_folder_path)
            else:
                self.control_count()

            # # 是否存在空白文件，不存在—>创建
            # if not exists(log_file):
            #     with open(log_file, 'w'):
            #         pass

            if fh_type == '1':
                logger.addHandler(self.init_fileHd(path=log_file_path))
            elif fh_type == '2':
                logger.addHandler(self.init_rotatingfileHd(path=log_file_path))
            elif fh_type == '3':
                logger.addHandler(self.init_timerotatingfileHd(path=log_file_path))
        return logger

    def init_streamHd(self):
        """
        流句柄
        """
        stream_handler = StreamHandler(stdout)
        stream_handler.setLevel(self.stream_level)
        stream_handler.setFormatter(LOG_FORMAT)
        return stream_handler

    def init_fileHd(self, path):
        """
        文件句柄1
        :param path: 输出的log文件路径
        """
        file_handler = FileHandler(path, encoding=LOG_ENCODING)
        file_handler.setLevel(self.save_level)
        file_handler.setFormatter(LOG_FORMAT)
        return file_handler

    def init_rotatingfileHd(self, path):
        """
        文件句柄2：按文件大小分割日志文件
        """
        rotfile_handler = RotatingFileHandler(
            filename=path,
            mode='a',  # 追加
            maxBytes=256 * 1024 * 1024,  # 当日志大小超过这个限制时会创建一个新的日志文件，然后再继续写入日志 256MB
            backupCount=self.backup_count,  # 允许备份文件数量上限
            encoding=LOG_ENCODING
        )
        rotfile_handler.setLevel(self.save_level)
        rotfile_handler.setFormatter(LOG_FORMAT)
        return rotfile_handler

    def init_timerotatingfileHd(self, path):
        """
        文件句柄3：按时间分割日志文件
        """
        timerot_file_handler = TimedRotatingFileHandler(
            filename=path,
            when=self.update_freq,  # 按时间间隔更新 'S': 秒 | 'M': 分钟 | 'H': 小时 | 'D': 天 | 'midnight': 每天的午夜
            interval=self.save_interval,  # 每隔1个when的时间间隔滚动一次日志文件
            backupCount=self.backup_count,  # 允许备份文件数量上限
            encoding=LOG_ENCODING
        )
        timerot_file_handler.setLevel(self.save_level)
        timerot_file_handler.setFormatter(LOG_FORMAT)
        return timerot_file_handler

    def control_count(self):
        """
        回滚文件
        """
        log_files = [f for f in listdir(self.log_folder_path) if f.startswith(LOG_PREFIX)]
        log_files.sort(key=lambda x: getctime(join(self.log_folder_path, x)))
        files_to_delete = len(log_files) - self.backup_count + 1
        for i in range(files_to_delete):
            file_to_delete = join(self.log_folder_path, log_files[i])
            remove(file_to_delete)
