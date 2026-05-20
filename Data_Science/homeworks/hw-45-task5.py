from series import Series
from dataframe import DataFrame
import numpy as np

data = DataFrame({"food": ["bacon", "pulled pork", "bacon",
"pastrami", "corned beef", "bacon",
"pastrami", "honey ham", "nova lox"],
"ounces": [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

# print(data)
# print()

meat_to_animal = {
"bacon": "pig",
"pulled pork": "pig",
"pastrami": "cow",
"corned beef": "cow",
"honey ham": "pig",
"nova lox": "salmon"
}

# data["animal"] = data["food"].map(meat_to_animal)
# print(data)

# def get_animal(x):
#     return meat_to_animal[x]

# print(data["food"].map(get_animal))

df = DataFrame({
    "A": [1, np.nan, 3],
    "B": [4, 5, np.nan]
})

print(df.map(lambda x: x*2))
print(df.map(lambda x: x*2, na_action="ignore"))

