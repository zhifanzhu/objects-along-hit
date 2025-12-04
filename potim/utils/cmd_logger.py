import logging
import hydra

FORMAT = (
    # "[%(asctime)s.%(msecs)d][%(filename)s:%(lineno)d]"
    # "[%(asctime)s][%(name)s][%(levelname)s]%(message)s"
    # "[%(asctime)s][%(filename)s:%(funcName)s #%(lineno)d][%(levelname)s] %(message)s"
    "[%(asctime)s][%(module)s: %(funcName)s() line %(lineno)d][%(levelname)s] %(message)s"
)
DATEFMT = "%Y-%m-%d %H:%M:%S"

def getLogger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # always do 'force'
    for h in logger.handlers[:]:
        logger.removeHandler(h)
        h.close()
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    logger.addHandler(h)
    logger.propagate = False
    return logger

def add_file_handler(logger, logfile: str):
    """ Add extra output file to logger """
    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)


def add_hydra_logfile(logger):
    """ Add hydra's log file to logger """
    _C_hydra = hydra.core.hydra_config.HydraConfig.get()
    log_path = _C_hydra.job_logging.handlers.file.filename
    add_file_handler(logger, log_path)