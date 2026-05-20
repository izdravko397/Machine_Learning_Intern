import my_logging

logger = my_logging.getLogger(__name__)

console_handler = my_logging.StreamHandler()
file_handler = my_logging.FileHandler("app.log", mode="a", encoding="utf-8")

logger.addHandler(console_handler)
logger.addHandler(file_handler)
print(logger.handlers)
# [
#   <StreamHandler <stderr> (NOTSET)>,
#   <FileHandler /Users/RealPython/Desktop/app.log (NOTSET)>
# ]

formatter = my_logging.Formatter(
   "{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

console_handler.setFormatter(formatter)

logger.warning("Look at my logger!")