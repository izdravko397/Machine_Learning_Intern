import re

exchange_rates = {
  "USD": 1.80277, "CHF": 2.03541, 
  "RON": 3.93385, "EUR": 1.95583, "GBP": 2.286
}

symbols = {'$': 'USD', '€': 'EUR', '£': 'GBP'}

# def convert_currency(text: str, exchange_rates, target_currency):
#     # monetary_values = re.findall(r'[$€£]\d+\.?\d*|\d+\.?\d*\s[$€£]?[\w]{0,3}', text)
#     monetary_value = re.findall(r'((?:[$€£]\d+(?:\.\d+)?)|(?:\d+(?:\.\d+)?\s?(?:[$€£]|[A-Za-z]{3})))', text)

#     for val in monetary_value:
#         if val[0] in symbols:
#             currency = val[0]
#             amount = float(val[1:])

#         elif val[-1] in symbols and ' ' not in val:
#             amount = float(val[:-1])
#             currency = val[-1]
        
#         elif val[-3:] in exchange_rates and ' ' not in val:
#             amount = float(val[:-3])
#             currency = val[-3:]

#         else:
#             amount, currency = val.split(' ')
#             amount = float(amount)

#         currency = symbols.get(currency, currency)
#         amount *= exchange_rates[currency]

#         if target_currency != 'BGN':
#             amount /= exchange_rates[target_currency]

#         text = text.replace(val, f'{int(amount)} {target_currency}')

#     return text


def convert_currency(text: str, exchange_rates, target_currency):
    def f(match_obj):
        val = match_obj.group(0)

        if val[0] in symbols:
            currency = val[0]
            amount = float(val[1:])

        elif val[-1] in symbols and ' ' not in val:
            amount = float(val[:-1])
            currency = val[-1]
        
        elif val[-3:] in exchange_rates and ' ' not in val:
            amount = float(val[:-3])
            currency = val[-3:]

        else:
            amount, currency = val.split(' ')
            amount = float(amount)

        currency = symbols.get(currency, currency)
        amount *= exchange_rates[currency]

        if target_currency != 'BGN':
            amount /= exchange_rates[target_currency]

        return f'{int(amount)} {target_currency}'

    return re.sub(r'((?:[$€£]\d+(?:\.\d+)?)|(?:\d+(?:\.\d+)?\s?(?:[$€£]|[A-Za-z]{3})))', f, text)

text = "Price: $25.00, valid today $25.00 only"
text = convert_currency(text, exchange_rates, "BGN")
# Price: 45 BGN valid today only
print(text)

text = "Price: 25.00$, valid today only"
text = convert_currency(text, exchange_rates, "BGN")
# Price: 45 BGN valid today only
print(text)

text = "Price: $2, valid today only"
text = convert_currency(text, exchange_rates, "BGN")
# Price: 45 BGN valid today only
print(text)

text = "Ivan Petrov: salary 2000 € for March 2024"
text = convert_currency(text, exchange_rates, "BGN")
# Ivan Petrov: salary 3910 BGN for March 2024
print(text)

text = "Ivan Petrov: salary 2000EUR for March 2024"
text = convert_currency(text, exchange_rates, "BGN")
# Ivan Petrov: salary 3910 BGN for March 2024
print(text)

text = "Ivan Petrov: salary 20 € for March 2024"
text = convert_currency(text, exchange_rates, "USD")
print(text)