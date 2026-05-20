def generate_deck():
    paints = ['H', 'S', 'C', 'D']
    cards_in_paint = ['A', '', 'J', 'Q', 'K'] 
    values = [num for num in range(1, 11)]
    val_len = len(values)
    paints_len = len(paints)

    paints_inx = 0
    cards_inx = 0
    value_inx = 0
    while True:
        paint = paints[paints_inx]
        card = cards_in_paint[cards_inx] if cards_inx != 1 else str(values[value_inx])
        value = values[value_inx]

        yield (paint + card, value)

        if card == 'K':
            paints_inx += 1
            if paints_inx == paints_len:
                break
            
            cards_inx = 0
            value_inx = 0
            continue

        if cards_inx == 1:
            value_inx += 1
            if value_inx == val_len:
                value_inx = -1
                cards_inx += 1
            continue

        cards_inx += 1
        
cards = generate_deck()
counter = 0
for card in cards:
    print(card)
    counter += 1
  
print(counter)
# (HA, 1)
# (H1, 1)