class Counter(dict):
    def __init__(self, iterable=None, **kwargs):
        super().__init__(**kwargs)
        if iterable is None:
            return
        
        for item in iterable:
            self[item] = self.get(item, 0) + 1

    def _sort(self):
        return sorted(self.items(), key=lambda x: x[1], reverse=True)
    
    def most_common(self, n):
        res = []
        keys_in_res = set()

        while n > 0:
            max_count_key = None
            max_count_val = float('-inf')
            for k, v in self.items():
                if v >= max_count_val and k not in keys_in_res:
                    max_count_key = k
                    max_count_val = v

            if max_count_key is None:
                break

            res.append((max_count_key, max_count_val))
            keys_in_res.add(max_count_key)
            n -= 1

        return res

        # return self._sort()[:n]
    
    def __str__(self):
        return f'Counter({dict(self._sort())})'

s = "hello world"
c = Counter(s)
print(c)

for e, count in c.most_common(3):
    print(e, "->", count)

# l -> 3
# o -> 2
# h -> 1
