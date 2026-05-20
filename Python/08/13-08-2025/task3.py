import inspect

def accepts(*valid_typs):
    def decorator(func): 
        def wrapper(*args):
            func_params_len = len(inspect.signature(func).parameters)
            args_len = len(args)

            if func_params_len != args_len:
                raise ValueError(f'Function expects {func_params_len} parameters, not {args_len}')
            
            for param, v_type in zip(args, valid_typs):
                if not isinstance(param, v_type):
                    raise TypeError(f'Input parameter {param} is of type {type(param).__name__}, expected type {v_type.__name__}')
                
            return func(*args)
        return wrapper
    return decorator

@accepts(int,int)
def bar(low,high):
    return 1.2

print(bar(1, 4, 5))