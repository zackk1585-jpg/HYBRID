import time

last_score = 0
total_spent = 0
total_earned = 0

def update_score(new_score):
    global last_score, total_earned

    delta = new_score - last_score
    if delta > 0:
        total_earned += delta

    last_score = new_score
    return delta

def register_shot(cost):
    global total_spent
    total_spent += cost

def get_profit():
    return total_earned - total_spent

def should_spend(bank):
    profit = get_profit()

    if profit > 50:
        return "aggressive"
    elif profit > 0:
        return "normal"
    else:
        return "conservative"