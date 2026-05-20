import hashlib, re

# def anonimyze(text: str):
#     pattern = r'\d{10}'
#     egn_list = re.findall(pattern, text)
#     anonimyzed_list = []

#     for data in egn_list:
#         to_sha1 = hashlib.sha1(data.encode(encoding='utf-8'))
#         anonimyzed_list.append(to_sha1.hexdigest()[-10:])

#     for egn, an_egn in zip(egn_list, anonimyzed_list):
#         text = text.replace(egn, an_egn)

#     return text

def anonimyze(text: str):
    def f(match_obj):
        egn = match_obj.group(2)
        to_sha1 = hashlib.sha1(egn.encode(encoding='utf-8'))
        
        return f'{match_obj.group(1)}{to_sha1.hexdigest()[-10:]}'

    return re.sub(r'(ЕГН:?\s+)(\d{10})', f, text)


text = "Иван Драганов, ЕГН:    9903142412 от град  9903142413 Пловдив"
anon_text = anonimyze(text)
print(anon_text)
# Иван Драганов, ЕГН 9d1a35fea8 от град Пловдив