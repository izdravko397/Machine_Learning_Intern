from cut_and_qcut import cut, qcut, Categorical
from series import Series
from get_dummies import get_dummies

# -------------- from codes ---------------
# fruits = ['apple', 'orange', 'apple', 'apple'] * 2
# ser = Series(fruits)
# print(ser)

# fruit_cat = ser.astype('category')
# print(fruit_cat)

# c = fruit_cat.array
# print(c)
# print(c.categories)
# print(c.codes)
# print(dict(enumerate(c.categories)))

# my_categories = Categorical(['foo', 'bar', 'baz', 'foo', 'bar'])
# print(my_categories)

# categories = ['foo', 'bar', 'baz']
# codes = [0, 1, 2, 0, 0, 1]

# my_cats_2 = Categorical.from_codes(codes, categories)
# print(my_cats_2)

# my_cats_2.as_ordered()
# print(my_cats_2)

# ordered_cat = Categorical.from_codes(codes, categories, ordered=True)
# print(ordered_cat)

#-------------- category Series ---------------
# s = Series(['a', 'b', 'c', 'd'] * 2, dtype='category')
# print(s.value_counts())
# print(s)

# s = Series(['a', 'b', 'c', 'd'] * 2)

# cat_s = s.astype('category')
# print(cat_s)
# print(cat_s.cat.codes)
# print(cat_s.cat.categories)

# print(cat_s.value_counts())

# actual_categories = ['a', 'b', 'c', 'd', 'e']
# cat_s2 = cat_s.cat.set_categories(actual_categories)
# print(cat_s2)
# print(cat_s2.cat.value_counts())

# cat_s2 = cat_s2.cat.remove_unused_categories()
# print(cat_s2)
# print(cat_s2.cat.value_counts())


# print(get_dummies(cat_s))