import my_logging


logger1 = my_logging.getLogger('app')
logger2 = my_logging.getLogger('app.module')
logger3 = my_logging.getLogger('app.module.submodule')

handler = my_logging.StreamHandler()
formatter = my_logging.Formatter('{asctime} -- {levelname} -- {name}', style='{')
handler.setFormatter(formatter)

logger1.addHandler(handler)
logger2.addHandler(handler)
logger3.addHandler(handler)

# logger1.propagate = False
# logger2 = my_logging.getLogger('app.module')
logger2.propagate = False
logger3.warning('warning')


# logger = my_logging.getLogger(__name__)
# print(logger.level)


# print(logger)


# print(logger.parent)