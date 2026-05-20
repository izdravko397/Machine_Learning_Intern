def create_object_array(class_name, data: str):
    initialize_class = f'class {class_name}:\n'
    data_lines = data.split('\n')
    initialize_class += f'    def __init__(self, {data_lines[0]}):\n'

    type_hints = []
    for p in data_lines[0].split(','):
        p_name, *type_hint = p.split(': ')
        type_hints.append(eval(type_hint[0]) if type_hint else None)

        initialize_class += f'        self.{p_name} = {p_name}\n'

    initialize_class += '    def __repr__(self):\n' \
    '        return f"{self.__class__.__name__}({", ".join(f"{key}={val}" for key, val in self.__dict__.items())})"'

    namespace = {}
    exec(initialize_class, namespace)
    executor_class = namespace[class_name]

    res = []
    for i in range(1, len (data_lines)):
        args = data_lines[i].split(',')

        for i in range(len(args)):
            if type_hints[i]:
                args[i] = type_hints[i](args[i])

        res.append(executor_class(*args))
    return res

def make_init(fields):
    def __init__(self, *args):
        for name, value in zip(fields, args):
            self.__dict__[name] = value
    return __init__

def create_object_array(class_name, data: str):
    data_lines = data.split('\n')

    args = []
    type_hints = []
    for arg in data_lines[0].split(','):
        arg_name, *type_hint = arg.split(': ')
        args.append(arg_name)
        type_hints.append(eval(type_hint[0]) if type_hint else None)

    new_class = type(class_name, (object,), {
        '__init__': make_init(args),
        '__repr__': lambda x: f'{class_name}({", ".join(f"{k}={v}" for k, v in x.__dict__.items())})'
    })

    res = []
    for i in range(1, len (data_lines)):
        params = data_lines[i].split(',')

        for i in range(len(args)):
            if type_hints[i]:
                params[i] = type_hints[i](params[i])

        res.append(new_class(*params))
    return res

data = """name: str,age: int
Ivan,12
Dragan,15
Albena,25"""
    
employees = create_object_array("Employee", data)
e = employees[0]
age_type = type(e.age) # int
print(age_type)


# employees = create_object_array("Employee", data)
print(employees)

# for e in employees:
#     print(e)
  
# Empployee(name=Ivan, age=12)
# Empployee(name=Dragan, age=15)
# Empployee(name=Albena, age=25)

# class Employee:
#   def __init__(self, name: str, age: int):
#     self.name = name
#     self.age = age