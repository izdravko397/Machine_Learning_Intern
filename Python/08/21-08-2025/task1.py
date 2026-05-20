import logging
import my_logging

def show_only_debug(record):
    return record['levelname'] == "DEBUG"

logger = my_logging.getLogger(__name__)
print(logger.getEffectiveLevel())

formatter = my_logging.Formatter('%(levelname)s', style='%')
formatter2 = my_logging.Formatter('{asctime}:{levelname}:{name}:{message}!!!!!!!v2')

handler = my_logging.StreamHandler(formatter)
handler.setLevel(my_logging.WARNING)

handler2 = my_logging.StreamHandler(formatter2)
handler2.addFilter(show_only_debug)

# handler2.setLevel()
f_handler = my_logging.FileHandler(formatter=formatter2)

logger.addHandler(handler)
logger.addHandler(handler2)
logger.addHandler(f_handler)
logger.setLevel(10)
print(logger.getEffectiveLevel())
# print(logger.handlers)

logger.info('info')
logger.warning('warning')
logger.debug('debug')

# g = 'bb'
# my_logging.basicConfig(level='INFO', format="{asctime} - {levelname} - {message}")
# my_logging.info(f'info {g=}')
# my_logging.basicConfig(level='DEBUG')
#--------------------------------------------------
# my_logging.debug('debug')
# logging.basicConfig(level='INFO')
# logging.warning('da')
# logging.info('dadada')
# logging.basicConfig(level='DEBUG')
# logging.debug('oda')


# logging.warning("Remain calm!")
# logging.debug("This is a debug message")
# logging.info("This is an info message")


#---------------------------------------------------
# logger = my_logging.getLogger(__name__)

# # Create handlers
# c_handler = my_logging.StreamHandler()
# f_handler = my_logging.FileHandler('file.log')
# c_handler.setLevel(my_logging.WARNING)
# f_handler.setLevel(my_logging.ERROR)

# # Create formatters and add it to handlers
# c_format = my_logging.Formatter('%(name)s - %(levelname)s - %(message)s', style='%')
# f_format = my_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', style='%')
# c_handler.setFormatter(c_format)
# f_handler.setFormatter(f_format)

# # Add handlers to the logger
# logger.addHandler(c_handler)
# logger.addHandler(f_handler)

# logger.warning('This is a warning')
# logger.error('This is an error')