import my_logging

logger = my_logging.getLogger(__name__)
logger.setLevel("DEBUG")
formatter = my_logging.Formatter("{levelname} - {message}", style="{")

console_handler = my_logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = my_logging.FileHandler("app.log", mode="a", encoding="utf-8")
file_handler.setLevel("WARNING")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.debug("Just checking in!")

logger.warning("Stay curious!")

logger.error("Stay put!")
