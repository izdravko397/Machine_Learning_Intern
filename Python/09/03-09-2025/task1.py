# car and cat
# pop and prop
# ferret, ferry, and ferrari
# Any word ending in ious
# A whitespace character followed by a period, comma, colon, or semicolon
# A word longer than six letters
# A word without the letter e (or E)

import re
# car and cat
str1 = "my car"
str2 = "bad cats" #=> True

str3 = "camper"
str4 = "high art" #=> False

pattern = r'ca[tr]'
# if re.search(pattern, str4):
#     print(True)
# else:
#     print(False)


#=============================
# pop and prop

str1 = "pop culture"
str2 = "mad props" #=> True

str3 = "plop"
str4 = "prrrop" #=> False

pattern = r'p[r]?op'
# if re.search(pattern, str4):
#     print(True)
# else:
#     print(False)

#=============================
# ferret, ferry, and ferrari

str1 = "ferret"
str2 = "ferry" #=> True
str3 = "ferrari" #=> True

str4 = "ferrum"
str5 = "transfer A" #=> False

pattern = r'ferret|ferry|ferrari'
# if re.search(pattern, str5):
#     print(True)
# else:
#     print(False)


#=============================
# Any word ending in ious

str1 = "how delicious"
str2 = "spacious room" #=> True

str3 = "ruinous"
str4 = "consciousness" #=> False

pattern = r'\b\w+ious\b'
# if re.search(pattern, str4):
#     print(True)
# else:
#     print(False)

#=============================
# A whitespace character followed by a period, comma, colon, or semicolon

str1 = "bad punctuation ." #=> True

str2 = "escape the period" #=> False

pattern = r'\s[.,:;]'
# if re.search(pattern, str2):
#     print(True)
# else:
#     print(False)

#=============================
# A word longer than six letters

str1 = "Siebentausenddreihundertzweiundzwanzig" #=> True

str2 = "no"
str3 = "three small words" #=> False

pattern = r'\b\w{6,}\b'
# if re.search(pattern, str2):
#     print(True)
# else:
#     print(False)

#=============================
# "red platypus", "wobbling nest" => True
# "earth bed", "bedrøvet abe", "BEET" => False
# A word without the letter e (or E)

str1 = "red platypus"
str2 = "wobbling nest" #=> True

str3 = "earth bed"
str4 = "bedrøvet abe"
str5 = "BEET" #=> False

pattern = r'\b[^eE\s]+\b'
# if re.search(pattern, str5):
#     print(True)
# else:
#     print(False)