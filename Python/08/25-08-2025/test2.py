import my_logging
import sys

# handler = my_logging.StreamHandler(stream=sys.stdout)

# logger = my_logging.getLogger('package') # Create logger with name 'package'
# logger.addHandler(handler)

# child_logger = my_logging.getLogger('package.module')
# child_logger.addHandler(handler)

# child_logger.propagate = False

# child_logger.warning('This will be printed by the parent and child handlers.')

# logger = my_logging.getLogger('package') # Create logger with name 'package'
# root_logger = my_logging.getLogger()

# print(f"The logger's level is: {logger.level}") # 0 == logging.NOTSET
# print(f"The root logger's level is: {root_logger.level}") # 30 == logging.WARNING

# print(f"This logger's effective level is: {logger.getEffectiveLevel()}") # 30 == logging.WARNING

# import logging
# root = logging.getLogger()
# logger = logging.getLogger('app')
# logger2 = logging.getLogger('app.mod')
# # print(root.parent)
# # print(logger.parent)
# # print(logger.level)
# logger2.propagate = False
# print(logger2.level)
# print(logger2.getEffectiveLevel())

config = {
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
    },

    'handlers': {
        'console': {
            'class': 'StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'sys.stdout'
        },
        'file': {
            'class': 'FileHandler',
            'level': 'INFO',
            'formatter': 'detailed',
            'filename': 'app.log',
            'encoding': 'utf-8'
        },
        'error_file': {
            'class': 'FileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'errors.log',
            'encoding': 'utf-8',
            'filters': ['only_errors']
        }
    },

    'loggers': {
        'my.module': {
            'level': 'DEBUG',
            'handlers': ['console'],
            'propagate': False
        },
        'my.app': {
            'level': 'INFO',
            'handlers': ['file'],
            'propagate': False
        }
    },
}

my_logging.dictConfig(config)

logger = my_logging.getLogger('my.app')
logger.warning('warning')
print(logger.handlers)
print(logger.propagate)