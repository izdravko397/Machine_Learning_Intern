from dataframe import DataFrame
import json

def read_json(fname):
    with open(fname) as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = [data]

    return DataFrame(data)

# data = read_json("examples/example.json")
# print(data)
# to_json = data.to_json()
# print(to_json)
# print(type(to_json))
# print(data['a'])

# data = read_json("examples/example1.json")
# print(data)

data = read_json("examples/example2.json")
print(data)
print(data.to_json())

# data = read_json("examples/example3.json")
# print(data)


# data = read_json("examples/example4.json")
# print(data)
# print(data.to_json())