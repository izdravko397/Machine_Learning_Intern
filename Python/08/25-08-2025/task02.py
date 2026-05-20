import my_logging

my_logging.warning("Remain calm!")

my_logging.debug("This is a debug message")

my_logging.info("This is an info message")

my_logging.warning("This is a warning message")


my_logging.error("This is an error message")


my_logging.critical("This is a critical message")

my_logging.basicConfig(level=my_logging.DEBUG, format="{asctime} - {levelname} - {message}", style="{", datefmt="%Y-%m-%d %H:%M")
my_logging.debug("This will get logged.")