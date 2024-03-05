import logging

def create_dir(path):
    """Créer le chemin de dossier 'path' s'il n'existe pas"""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def create_logger(name, path, format="%(asctime)s - %(message)s", level='INFO', mode='w'):
    """Créer un logger"""
    logger = logging.getLogger(name)
    handle = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format)
    handle.setFormatter(formatter)
    logger.addHandler(handle)
    logger.setLevel(level)
    return logger