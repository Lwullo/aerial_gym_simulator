import logging
from logging import Logger

BLUE_LEVEL = 25
GREEN_LEVEL = 26
ORANGE_LEVEL = 27
logging.addLevelName(BLUE_LEVEL, "BLUE")
logging.addLevelName(GREEN_LEVEL, "GREEN")
logging.addLevelName(ORANGE_LEVEL, "ORANGE")


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"
    orange = "\x1b[38;5;208m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    white = "\x1b[37m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    reset = "\x1b[0m"
    format = (
        "[%(relativeCreated)d ms][%(name)s] - %(levelname)s : %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: cyan + format + reset,
        logging.INFO: white + format + reset,
        BLUE_LEVEL: blue + format + reset,
        GREEN_LEVEL: green + format + reset,
        ORANGE_LEVEL: orange + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomLogger(Logger):
    def __init__(self, logger_name):
        # call superclass
        super().__init__(logger_name)
        self.setLevel(logging.INFO)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        self.ch.setFormatter(CustomFormatter())
        self.addHandler(self.ch)

    def setLoggerLevel(self, level) -> None:
        self.setLevel(level)
        self.ch.setLevel(level)

    def print_example_message(self):
        self.debug("A Debug message will look like this")
        self.info("An Info message will look like this")
        self.blue("A Blue message will look like this")
        self.green("A Green message will look like this")
        self.orange("An Orange message will look like this")
        self.warning("A Warning message will look like this")
        self.error("An Error message will look like this")
        self.critical("A Critical message will look like this")

    def blue(self, message, *args, **kwargs):
        if self.isEnabledFor(BLUE_LEVEL):
            self._log(BLUE_LEVEL, message, args, **kwargs)

    def green(self, message, *args, **kwargs):
        if self.isEnabledFor(GREEN_LEVEL):
            self._log(GREEN_LEVEL, message, args, **kwargs)

    def orange(self, message, *args, **kwargs):
        if self.isEnabledFor(ORANGE_LEVEL):
            self._log(ORANGE_LEVEL, message, args, **kwargs)
