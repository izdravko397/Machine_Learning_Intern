from index import *
import pandas as pd


class PeriodIndex(Index):
    def __init__(self, data, freq, name=''):
        if not all(isinstance(x, Period) for x in data):
            raise TypeError('Invalid type for data, must be Periods')
        
        if not isinstance(data, np.ndarray):
            data = np.fromiter((x for x in data), np.object_)

        self._data = data
        self.freq = freq
        self.name = name
        self._dtype = f'period[{freq}]'

    def is_monotonic(self):
        return all(self._data.end[i] <= self._data.start[i + 1] for i in range(len(self) - 1))

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for p in self._data:
            yield p.value

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return f'PeriodIndex([{", ".join(self)}]'\
            f', dtype="{self._dtype}"'



def period_range(start=None, end=None, periods=None, freq: str =None, name=''):
    if freq is None:
        raise ValueError('"freq" required')
    
    if start is None and end is None:
        raise ValueError("Must specify start or end or both")

    if ((start is None and end is not None) or (start is not None and end is None)) and periods is None:
        raise ValueError("Must specify (start or end) and periods")
   
    if start is not None:
        start = Period(start, freq)
    if end is not None:
        end = Period(end, freq)
    
    n, freq_type, ex1, ex2 = parse_freq(freq)
    offset = get_offset(n=n, offset_type=freq_type, extra1=ex1, extra2=ex2)
    n = 1 if isinstance(offset, (Minute, Hour, Day, WeekEnd)) else 2

    if periods is not None:
        if start is not None and end is not None:
            raise ValueError("Can not specify start, end and periods")
    
        if not isinstance(periods, int) or periods <= 0:
            raise ValueError("periods must be a positive integer")

        is_add = start is not None
        first = start or end        
        data = [first]
        for _ in range(1, periods):
            previous_p_ts = pd.Timestamp(data[-1].value)
            if is_add:
                offset *= n
                next_period_ts = previous_p_ts + offset
            else:
                next_period_ts = previous_p_ts - offset

            p_str = Period.str_repr[offset.code](next_period_ts)
            data.append(Period(p_str, freq))
        
        if start is None:
            data.reverse()

        return PeriodIndex(data, freq, name)

    if start is None or end is None:
        raise ValueError("Must specify start and end if no period is submitted")
    
    offset *= n
    data = [start]
    current = start.end
    if freq.startswith('Q'):
        end = end.end
    else:
        end = end.start

    while current < end:
        previous_p_ts = data[-1].start
        next_period_ts = previous_p_ts + offset
        p_str = Period.str_repr[offset.code](next_period_ts, ex1, ex2)
        
        period = Period(p_str, freq)
        data.append(period)
        current = period.end

    return PeriodIndex(data, freq, name)


class Period:
    str_repr = {
        'A':   lambda x, *_: f'{x.year}',
        'M':   lambda x, *_: f'{x.year}-{x.month}',
        'D':   lambda x, *_: f'{x.year}-{x.month}-{x.day}',
        'B':   lambda x, *_: f'{x.year}-{x.month}-{x.day}',
        'W':   lambda x, *_: f'{x.year}-{x.month}-{x.day}',
        'H':   lambda x, *_: f'{x.year}-{x.month}-{x.day} {x.hour}:0',
        'T':   lambda x, *_: f'{x.year}-{x.month}-{x.day} {x.hour}:{x.minute}',
        'min': lambda x, *_: f'{x.year}-{x.month}-{x.day} {x.hour}:{x.minute}',
        'Q':   lambda x, month, *_: f"{x.year + (x.month > month)}Q{((x.month - month - 1) % 12)//3 + 1}"
    }

    get_val = {
        'A':   lambda x: x.year,
        'M':   lambda x: x.month,
        'D':   lambda x: x.day,
        'B':   lambda x: x.day,
        'H':   lambda x: x.hour,
        'T':   lambda x: x.minute,
        'min': lambda x: x.minute,
        'W':   lambda x: x.weekday(),
    }

    def __init__(self, value, freq):
        self._start = None
        self._end = None
        self.value = value
        self.freq = freq
        self.offset = get_offset(freq)

        self._val_ts = value
        if not isinstance(value, pd.Timestamp):
            self._val_ts = to_datetime(value)

    @property
    def start(self):
        if self._start is not None:
            return self._start
        
        if self.offset.code == 'Q':
            year, qu = self.value.split('Q')
            f_start_year = Period(year, f'A-{months[self.offset.month - 1]}').start
            qu = int(qu)

            if qu > 1:
                self._start = f_start_year + (self.offset * (qu - 1)) + Day()
            else:
                self._start = f_start_year

        elif self.offset.code == 'W':
            self._start = self._val_ts - self.offset + Day()

        elif self.offset.code == 'A' and self.freq != 'A-DEC':
            start = self._val_ts
            end = self.value + YearEnd()
            current_month = pd.Timestamp(f'{start.year}-{self.offset.month}')

            begin_delta = current_month - start
            end_delta = end - current_month
            total_delta = begin_delta + end_delta

            ts = current_month - total_delta
            self._start = pd.Timestamp(f'{ts.year}-{ts.month}') + MonthEnd() + Day()
        else:
            self._start = self._val_ts

        return self._start
    
    @property
    def end(self):
        if self._end is not None:
            return self._end
        
        if self.offset.code == 'Q':
            end = self.start + self.offset
        else:
            end = self._val_ts + self.offset
    
        self._end = end
        return end

    def asfreq(self, freq, how='end'):
        if how == 'start':
            date = self.start
        elif how == 'end':
            date = self.end
        else:
            raise ValueError('Invalid "how"')

        _, freq_type, month, _ = parse_freq(freq)

        if freq_type == 'A' and month < self.start.month:
            date += YearEnd(2)

        return Period(self.str_repr[freq_type](date), freq)

    def __add__(self, other):
        if isinstance(other, Period):
            raise TypeError("unsupported operand type(s) for +: 'Period' and 'Period'")
        
        offset = self.offset * other
        ts_val = offset + self.start
        
        code = self.offset.code
        new_val = self.str_repr[code](ts_val)

        if code == 'Y':
            new_val = str(int(new_val) + 1)
        
        elif code == 'M':
            old = new_val.split('-')[1]
            new_val = new_val.replace(old, str(int(old) + 1))

        return Period(new_val, self.freq)
    
    def __sub__(self, other):
        if isinstance(other, Period):
            if self.freq != other.freq:
                raise ValueError('Not match freq')
            
            sf = self.get_val[self.offset.code](self.start)
            oth = self.get_val[self.offset.code](other.start)

            n = abs(sf - oth)
            return self.offset.__class__(n)

        offset = self.offset * other
        ts_val = offset - self.start

        new_val = self.str_repr[self.offset.code](ts_val)
        return Period(new_val, self.freq)

    def __repr__(self):
        return f'Period("{self.value}", "{self.freq}")'
    



























# p = Period("2011", freq="A-DEC")
# print(p)
# print(p.offset)
# p2 = p + 5
# print(p2)
# print(p2.start)

# print(p - 2)

# print(Period("2014", freq="A-DEC") - p)
# print(Period("2014", freq="A-DEC") + p)

# print(p.asfreq("M", how="start"))
# print(p.asfreq("M", how="end"))
# print(p.asfreq("M"))

# p = Period("2011", freq="A-JUN")
# print(p)
# print(p.offset)
# print(p.start)

# print(p.asfreq("M", how="start"))
# print(p.asfreq("M", how="end"))

# p = Period("Aug-2011", "M")
# print(p)
# print(p.start)
# print(p.asfreq('H'))

# print(repr(p.asfreq("A-JUN")))


# periods = period_range("2006", periods=3, freq="A-DEC")
# print(periods)
# print(periods._data)

# periods = period_range(end="2006", periods=3, freq="A-DEC")
# print(periods)
# print(periods._data)


# periods = period_range("2006", "2009", freq="A-DEC")
# print(periods)
# print(periods._data)

