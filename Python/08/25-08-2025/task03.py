import my_logging

my_logging.basicConfig(
    filename="app.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)

my_logging.warning("Save me!")

donuts = 5
guests = 0
try:
    donuts_per_guest = donuts / guests
except ZeroDivisionError:
    # my_logging.error("DonutCalculationError", exc_info=True)
    my_logging.exception("DonutCalculationError")