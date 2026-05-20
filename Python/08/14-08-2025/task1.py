def create_object_array(class_name, data: str):
    initialize_class = f'class {class_name}:\n'
    data_lines = data.split('\n')
    constructor = data_lines[0].split(',')
    initialize_class += f'    def __init__(self, {', '.join(p for p in constructor)}):\n'

    for p in constructor:
        initialize_class += f'        self.{p} = {p}\n'

    initialize_class += '    def __repr__(self):\n' \
    '        return f"{self.__class__.__name__}({", ".join(f"{key}={val}" for key, val in self.__dict__.items())})"'

    namespace = {}
    exec(initialize_class, namespace)
    executor_class = namespace[class_name]

    return [executor_class(*data_lines[i].split(',')) for i in range(1, len(data_lines))]

data = """name,age
Ivan,12
Dragan,15
Albena,25"""

employees = create_object_array("Employee", data)
# print(employees)

for e in employees:
    print(e)
  
# Empployee(name=Ivan, age=12)
# Empployee(name=Dragan, age=15)
# Empployee(name=Albena, age=25)