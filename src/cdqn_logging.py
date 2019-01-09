import logging
import os

cdqn_logger = logging.getLogger()
cdqn_logger.setLevel(logging.DEBUG)

cdqn_logformat = '%(asctime)s :: %(levelname)s :: %(message)s'
cdqn_formatter = logging.Formatter(cdqn_logformat)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(cdqn_formatter)
cdqn_logger.addHandler(stream_handler)


def create_runlogger(run_number: int = 0, log_path: str = ''):
    logger_name = 'alien_gym.run.{}'.format(run_number)
    run_logger = logging.getLogger(logger_name)
    run_logger.addHandler(stream_handler)

    formatter = logging.Formatter(cdqn_logformat)
    file_handler = logging.FileHandler(os.path.join(log_path, 'activity.log'), 'a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    run_logger.addHandler(file_handler)
    return logger_name, run_logger
