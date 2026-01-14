import os
import random

DATASET_PATH = "./market_basket_dataset.csv"
TARGET_BILLNO = 5000

ITEMS = [
    "Apples", "Bananas", "Bread", "Butter", "Cereal", "Cheese",
    "Chicken", "Coffee", "Eggs", "Juice", "Milk", "Onions",
    "Oranges", "Pasta", "Potatoes", "Sugar", "Tea", "Tomatoes", "Yogurt"
]

CLUSTERS = [
    {"Bread", "Butter", "Coffee", "Sugar"},
    {"Milk", "Cereal", "Bananas"},
    {"Chicken", "Potatoes", "Onions", "Tomatoes"},
    {"Yogurt", "Apples", "Bananas"},
    {"Tea", "Coffee", "Sugar"},
    {"Cheese", "Pasta", "Tomatoes"},
    {"Oranges", "Apples", "Bananas"},
]

def parse_dataset(path):
    max_bill = 0
    price_stats = {}
    with open(path, "r") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            s = line.strip()
            if not s:
                continue
            parts = s.split(";")
            if len(parts) < 5:
                continue
            try:
                bill = int(parts[0])
                if bill > max_bill:
                    max_bill = bill
            except ValueError:
                pass
            item = parts[1]
            try:
                price = float(parts[3])
            except ValueError:
                continue
            if item not in price_stats:
                price_stats[item] = [price, price]
            else:
                lo, hi = price_stats[item]
                if price < lo:
                    lo = price
                if price > hi:
                    hi = price
                price_stats[item] = [lo, hi]
    # convert lists to tuples
    price_stats = {k: (v[0], v[1]) for k, v in price_stats.items()}
    return max_bill, price_stats

def sample_basket(rng):
    target_size = rng.randint(3, 8)
    base = set(rng.choice(CLUSTERS))
    candidates = [i for i in ITEMS if i not in base]
    if len(base) < target_size:
        extra = rng.sample(candidates, target_size - len(base))
        base.update(extra)
    else:
        base = set(rng.sample(list(base), target_size))
    return list(base)

def price_for(item, rng, stats):
    default_range = (1.0, 9.99)
    lo, hi = stats.get(item, default_range)
    span = hi - lo
    lo_adj = max(0.99, lo - (span * 0.10 if span > 0 else 0.25))
    hi_adj = min(9.99, hi + (span * 0.10 if span > 0 else 0.25))
    p = rng.uniform(lo_adj, hi_adj)
    return round(p, 2)

def main():
    path = DATASET_PATH
    rng = random.Random()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    max_bill, stats = parse_dataset(path)
    if max_bill >= TARGET_BILLNO:
        print(f"Dataset already reaches BillNo {max_bill}, nothing to append.")
        return
    appended = 0
    with open(path, "a") as out:
        for bill in range(max_bill + 1, TARGET_BILLNO + 1):
            cust_id = rng.randint(10000, 99999)
            basket = sample_basket(rng)
            rng.shuffle(basket)
            for item in basket:
                qty = rng.randint(1, 5)
                price = price_for(item, rng, stats)
                out.write(f"{bill};{item};{qty};{price:.2f};{cust_id};\n")
                appended += 1
    print(f"Appended {appended} lines for BillNo {max_bill+1}..{TARGET_BILLNO}.")

if __name__ == "__main__":
    main()