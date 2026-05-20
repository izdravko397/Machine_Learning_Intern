from .handlers import StreamHandler, level_val
from .filter import Filter

# def getLogger(name):
#     from . import getLogger
#     return getLogger(name)

class Record:
    def __init__(self, levelname, message, name):
        self.levelname = levelname
        self.message = message
        self.name = name

_hierarchy_levels = []
class Logger:
    def __init__(self, level=30, qualname='root', propagate=True):
        self.qualname = qualname
        self.level = level if qualname == 'root' else 0
        self.base_handler = StreamHandler()
        self._handlers = []
        self.propagate = False if qualname == 'root' else propagate
        self.filters = []
        self.parent = None

    @property
    def handlers(self):
        return '[\n' + '\n'.join(str(h) for h in self._handlers) + '\n]'
    
    def _check_level(self, log_level):
        return level_val[log_level] < self.level

    def _log(self, level, msg, exc_info, record=None):
        if self._check_level(level):
            return
        
        record = record or Record(level, msg, self.qualname)

        current_logger = self
        while True:
            if not all(f.filter(record) for f in current_logger.filters):
                break

            for h in current_logger._handlers:
                h.print_logger(record, exc_info)

            if current_logger.qualname == 'root' and not current_logger._handlers:
                current_logger.base_handler.print_logger(record, exc_info)
                break

            elif current_logger.propagate:
                if current_logger.parent.qualname == 'root' and current_logger._handlers:
                    break

                current_logger = current_logger.parent
            else:
                break

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
        if isinstance(level, str):
            level = level_val.get(level)
            if level is None:
                raise ValueError('invalid level')

        self.level = level

    def getEffectiveLevel(self):
        _hierarchy_levels.append(self.level)

        if not (parent_logger := self.parent):
            res = max(_hierarchy_levels)
            _hierarchy_levels.clear()
            return res
        
        return parent_logger.getEffectiveLevel()
    
    def addFilter(self, filter):
        class Func_filter:
            def __init__(self, filter):
                self.filter = filter

        if not isinstance(filter, Filter):
            filter = Func_filter(filter)

        self.filters.append(filter)
    
    def __str__(self):
        return f'<{'RootLogger' if self.qualname == 'root' else 'Logger'} {self.qualname} ({self.level})>'
    
