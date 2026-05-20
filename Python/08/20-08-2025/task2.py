import logging
import logging.config
import os
import sys

script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, 'log-config.ini')

def read_file(file):
    fields = {
        'loggers' : {},
        'handlers' : {},
        'formatters' : {}
    }

    get_keys = None
    get_next_lines = False
    type = None
    key = None

    with open(file) as config:
        for line in config:
            line = line.strip()

            if line in {'[loggers]', '[handlers]', '[formatters]'}:
                get_keys = line[1:-1]
                continue

            if get_keys:
                types = line.split('=')

                for t in types[1].split(','):
                    fields[get_keys].update({f'{t}': {}})

                get_keys = None
                continue
            
            if line.startswith('['):
                type, key = line.split('_')
                type = type[1:] + 's'
                key = key[:-1]
                get_next_lines = True
                continue

            if get_next_lines:
                if not line:
                    get_next_lines = False
                    continue

                k, v = line.split('=')
                fields[type][key][k] = v

    return fields

def logging_fileConfig(fname):
    config_dict = read_file(fname)

    formatters = {}
    for f_name, params in config_dict['formatters'].items():
        formatters[f_name] = logging.Formatter(params['format'])

    handlers = {}
    for h_name, params in config_dict['handlers'].items():
        handler_class = getattr(logging, params['class'])
        h_instance = handler_class(*eval(params.get('args', '()')))
        h_instance.setLevel(getattr(logging, params['level']))
        h_instance.setFormatter(formatters[params['formatter']])

        handlers[h_name] = h_instance

    for l_name, params in config_dict['loggers'].items():
        logger = logging.getLogger(params.get('qualname', ''))

        logger.addHandler(handlers[params['handlers']])
        logger.setLevel(params['level'])

        if propagate := params.get('propagate'):
            val = int(propagate)
            logger.propagate = val

# logging.config.fileConfig(config_path, disable_existing_loggers=False)
logging_fileConfig(fname=config_path)

logger = logging.getLogger(__name__)

logger.warning('This is a warning')
logger.error('This is an error')

# --------------------------------------------------------
# import logging

# logger = logging.getLogger(__name__)

# # Create handlers
# c_handler = logging.StreamHandler()
# f_handler = logging.FileHandler('file.log')
# c_handler.setLevel(logging.WARNING)
# f_handler.setLevel(logging.ERROR)

# # Create formatters and add it to handlers
# c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
# f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# c_handler.setFormatter(c_format)
# f_handler.setFormatter(f_format)

# # Add handlers to the logger
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)

# logger.warning('This is a warning')
# logger.error('This is an error')