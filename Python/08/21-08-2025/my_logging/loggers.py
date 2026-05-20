import datetime
from .handlers import StreamHandler, level_val
from .filter import Filter

def get_root():
    from . import root
    return root

class Record:
    def __init__(self, levelname, message, name):
        self.levelname = levelname
        self.message = message
        self.name = name


class Logger:
    def __init__(self, level='WARNING', qualname='root', propagate=True):
        self.level = level
        self.qualname = qualname
        self.base_handler = StreamHandler()
        self._handlers = []
        self.propagate = propagate
        self.filters = []
        self.parent = None

    @property
    def handlers(self):
        return '[\n' + '\n'.join(str(h) for h in self._handlers) + '\n]'
    
    def _check_level(self, log_level):
        return level_val[log_level] < level_val[self.level]

    def _log(self, level, msg, exc_info, record=None):
        if self._check_level(level):
            return
        
        record = record or Record(level, msg, self.qualname)

        for f in self.filters:
            if not f.filter(record):
                return

        if self._handlers:
            for h in self._handlers:
                record.asctime = datetime.datetime.now().strftime(h.formatter.datefmt)
                h.print_logger(record, exc_info)

        elif self.qualname == 'root':
            self.base_handler.print_logger(record, exc_info)

        elif self.propagate:
            if self.parent is None:
                self.parent = get_root()
            self.parent._log(level=level, msg=msg, record=record, exc_info=exc_info)

    def debug(self, msg, exc_info=False):
        self._log('DEBUG', msg, exc_info)

    def info(self, msg, exc_info=False):
        self._log('INFO', msg, exc_info)

    def warning(self, msg, exc_info=False):
        self._log('WARNING', msg, exc_info)

    def error(self, msg, exc_info=False):
        self._log('ERROR', msg, exc_info)

    def critical(self, msg, exc_info=False):
        self._log('CRITICAL', msg, exc_info)

    def addHandler(self, handler):
        self._handlers.append(handler)

    def setLevel(self, level):
        if isinstance(level, int):
            for k, v in level_val.items():
                if level == v:
                    level = k

        self.level = level

    def getEffectiveLevel(self):
        return level_val[self.level]
    
    def addFilter(self, filter):
        class Func_filter:
            def __init__(self, filter):
                self.filter = filter

        if not isinstance(filter, Filter):
            filter = Func_filter(filter)

        self.filters.append(filter)
    
    def __str__(self):
        return f'<{'RootLogger' if self.qualname == 'root' else 'Logger'} {self.qualname} ({self.level})>'