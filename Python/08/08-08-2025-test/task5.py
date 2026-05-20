suits = "HSCD"
type_card = ['A', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

deck = [(s + type_card[i], int(type_card[i]) if 0 < i < 11 else 1 if not i else 10) for s in suits for i in range(len(type_card))]
print(deck)