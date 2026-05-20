import my_logging

# my_logging.basicConfig(
#     filename="app.log",
#     encoding="utf-8",
#     filemode="a",
#     format="{asctime} - {levelname} - {message}",
#     style="{",
#     datefmt="%Y-%m-%d %H:%M",
# )

# donuts = 5
# guests = 0


# try:
#     donuts_per_guest = donuts / guests
# except ZeroDivisionError:
#     my_logging.exception('az sum dobre')
    # my_logging.error("DonutCalculationError", exc_info=True)

#------------------------------------------

def show_only_debug(record):
    return record.levelname == "DEBUG"

def show_only_error(record):
    return record.levelname == "ERROR"

logger = my_logging.getLogger(__name__)
#----------------------------------------------------------------
# logger2 = my_logging.getLogger('logger2')

# logger2.parent = logger
# logger.propagate = False

# logger2.warning('warning')

#--------------------------------------------------

logger.setLevel("DEBUG")
formatter = my_logging.Formatter("{levelname} - {message}", style="{")

class Level_filter(my_logging.Filter):
    def __init__(self, level, name=''):
        super().__init__(name)
        self.level = level

    def filter(self, record):
        return record.levelname == self.level

# logger.addFilter(my_logging.Filter('__'))
# logger.addFilter(Level_filter('ERROR'))

console_handler = my_logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)
console_handler.addFilter(show_only_debug)
# print(console_handler)
logger.addHandler(console_handler)

file_handler = my_logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel("WARNING")
file_handler.setFormatter(formatter)
file_handler.addFilter(Level_filter('WARNING'))
# print(file_handler)
logger.addHandler(file_handler)

logger.debug("Just checking in!")


logger.warning("Stay curious!")


logger.error("Stay put!")

# logger.debug("Just checking in!")
# logger.info("Just checking in, again!")


# console_handler = my_logging.StreamHandler()
# file_handler = my_logging.FileHandler("app.log", mode="a", encoding="utf-8")

# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# formatter = my_logging.Formatter(
#    "{asctime} - {levelname} - {message}",
#     style="{",
#     datefmt="%Y-%m-%d %H:%M",
# )

# console_handler.setFormatter(formatter)

# logger.warning("Watch out!")