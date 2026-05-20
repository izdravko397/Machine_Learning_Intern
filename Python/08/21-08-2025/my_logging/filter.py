class Filter:
    def __init__(self, name=''):
        self.name = name

    def filter(self, record):
        return record.name.startswith(self.name)