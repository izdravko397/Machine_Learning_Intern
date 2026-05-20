class ABC:
    def __new__(cls):
        if cls in ABC.__subclasses__() or cls is ABC:
            raise TypeError(f"Can't instantiate abstract class {cls.__name__}")
        
        no_implement = []
        mro = cls.mro()
        for i in range(1, len(mro) - 1):
            c = mro[i]
            print(dir(c))
            for name in filter(lambda x: callable(getattr(c, x)) and hasattr(getattr(c, x), '__dict__'), dir(c)):
                if getattr(c, name).__dict__.get('is_abstract') and name not in cls.__dict__:
                    no_implement.append(name)

        if no_implement:
           raise TypeError(f"Can't instantiate abstract class {cls.__name__}, with abstractmethods {', '.join(m for m in no_implement)}.")
                    
        return object.__new__(cls)
        
def abstractmethod(func):
    func.__dict__['is_abstract'] = True
    return func
        
class Stream(ABC):
  @abstractmethod
  def receive(self):
    pass
  
  @abstractmethod
  def send(self, msg):
    pass
  
  @abstractmethod
  def close(self):
    pass

class SocketStream(Stream):
    def receive(self):
        print("receive")

    def send(self, msg):
        print("send")

    def close(self):
        print("close")

s = SocketStream()
s.send('aa')