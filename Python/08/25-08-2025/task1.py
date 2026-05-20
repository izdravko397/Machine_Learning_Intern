import my_logging 

def show_only_debug(record):
    return record.levelname == "DEBUG"

def show_only_error(record):
    return record.levelname == "ERROR"

class Level_filter(my_logging.Filter):
    def __init__(self, level, name=''):
        super().__init__(name)
        self.level = level

    def filter(self, record):
        return record.levelname == self.level

logger = my_logging.getLogger(__name__)

logger.setLevel("DEBUG")
formatter = my_logging.Formatter("{levelname} - {message}", style="{")

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
#----------------------------------------------------------

import my_logging
import sys
from my_logging import Logger
import platform
import socket
from functools import partial


env: str = "development"
# log: Logger = my_logging.getLogger("app_name")
# log.setLevel(my_logging.DEBUG)

# format: str = f"%(asctime)s app_name {env}  %(name)s: %(message)s"
# formatter: my_logging.Formatter = my_logging.Formatter(format, style='%', datefmt="%b %d %H:%M:%S")

# stream_handler: my_logging.Handler = my_logging.StreamHandler(sys.stdout)
# stream_handler.setFormatter(formatter)
# log.addHandler(stream_handler)

# i: int = 0
# while i < 10:
#     log.debug(msg=f"Hello from my custom log {i}")
#     i += 1
#===========================================================================
class UptimeEndpointFilter(my_logging.Filter):
    def filter(self, record) -> bool:
        if "GET /up" in record.message:
            return False
        else:
            return True
        
# log: Logger = my_logging.getLogger("app_name")
# log.setLevel(my_logging.DEBUG)

# filter: UptimeEndpointFilter = UptimeEndpointFilter(name="uptime_filter")

# format = f"%(asctime)s app_name {env}  %(name)s: %(message)s"
# formatter: my_logging.Formatter = my_logging.Formatter(format, style='%', datefmt="%b %d %H:%M:%S")

# stream_handler: my_logging.Handler = my_logging.StreamHandler(sys.stdout)
# stream_handler.setFormatter(formatter)
# log.addHandler(stream_handler)
# log.addFilter(filter)

# i: int = 0
# while i < 10:
#     log.debug(msg=f"Hello from my custom log {i}")
#     log.debug(msg=f"GET /up")
#     i += 1
#==============================================

# log: Logger = my_logging.getLogger("app_name")
# log.setLevel(my_logging.DEBUG)

# #Filter
# filter: UptimeEndpointFilter = UptimeEndpointFilter(name="uptime_filter")
# # log.addFilter(filter=filter)

# #Formatter
# format = f"%(asctime)s app_name {env}  %(name)s: %(message)s"
# formatter: my_logging.Formatter = my_logging.Formatter(format, style='%', datefmt="%b %d %H:%M:%S")

# #Stream Handler
# stream_handler: my_logging.Handler = my_logging.StreamHandler(sys.stdout)
# stream_handler.setFormatter(formatter)
# log.addHandler(stream_handler)
# stream_handler.addFilter(filter=filter)

# #File Handler
# file_handler: my_logging.Handler = my_logging.FileHandler(
#     fname="mylogs.log", mode="a", encoding="utf-8"
# )
# log.addHandler(file_handler)
# log.addHandler(stream_handler)

# i: int = 0
# while i < 10:
#     log.debug(msg=f"Hello from my custom log {i}")
#     log.debug(msg=f"GET /up")
#     i += 1