def accepts(*valid_typs):
    def decorator(func): 
        def wrapper(*args):
            for param, v_type in zip(args, valid_typs):
                if not isinstance(param, v_type):
                    raise TypeError(f'Input parameter {param} is of type {type(param).__name__}, expected type {v_type.__name__}')
                
            return func(*args)
        return wrapper
    return decorator

def returns(valid_type):
    def decorator(func):
        def wrapper(*args):
            res = func(*args)
            if not isinstance(res, valid_type):
                raise TypeError(f'Output is of type {type(res).__name__}, expected type {valid_type.__name__}')
            
            return res
        return wrapper
    return decorator

@accepts(int,int)
@returns(float)
def bar(low,high):
    return 1.2

print(bar(1, 4))

from functools import wraps

def checktypes(func):
    @wraps(func)
    def wrapper(*args):
        type_hints = func.__annotations__

        for param, v_type in zip(args, type_hints.items()):
            if v_type[0] == 'return':
                break

            if not isinstance(param, v_type[1]):
                raise TypeError(f'Input parameter {param} is of type {type(param).__name__}, expected type {v_type[1].__name__}')
            
        res = func(*args)

        if output_type := type_hints.get('return'):
            if not isinstance(res, output_type):
                raise TypeError(f'Output is of type {type(res).__name__}, expected type {output_type.__name__}')
        
        return res
    return wrapper

@checktypes
def bar(low: int, high: int) -> float:
    return 1.2

print(bar(1, 2))