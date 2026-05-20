class ExchangeRate:
    def __init__(self, currencies):
        self.currencies_exchange_rates = currencies if not isinstance(currencies, list) else {key: value for key, value in currencies}

    def __getattr__(self, name):
        if name in self.currencies_exchange_rates:
            return self.currencies_exchange_rates[name]
        
        try:
            change, course1, course2 = name.split('_')
            if change != 'change' or course2 not in self.currencies_exchange_rates:
                raise 
        except Exception:
            raise AttributeError

        return Change(course1, course2, self.currencies_exchange_rates[course2], self.currencies_exchange_rates.get(course1))
    
    def __setattr__(self, name, value):
        if name == 'currencies_exchange_rates':
            # super().__setattr__(name, value)
            self.__dict__[name] = value
        else:
            self.currencies_exchange_rates[name] = value

class Change:
    def __init__(self, course1, course2, val2, val1=1):
        self.course1 = course1
        self.course2 = course2
        self.course2_val = val2
        self.course1_val = val1

    def __call__(self, *args):
        if self.course1 == 'lv':
            return args[0] / self.course2_val
        
        course1_to_lv = args[0] * self.course1_val
        lv_to_course2 = course1_to_lv / self.course2_val

        return lv_to_course2

rates_dict = {"usd": 1.69, "euro": 1.96, "gbp": 2.9}
# rates_dict = [('usd', 1.5), ('ud', 1.5), ('d', 1.5), ('sd', 1.5)]
rates = ExchangeRate(rates_dict)
print(rates.usd)

prive_lv = rates.change_lv_usd(1.5)
print(prive_lv)

prive_lv = rates.change_lv_euro(1)
print(prive_lv)

prive_usd = rates.change_euro_usd(1)
print(prive_usd)

rates.usd = 1.99
print(rates.usd)

rates.r = 1.20
print(rates.r)
print(rates.currencies_exchange_rates)
print(rates.__dict__)