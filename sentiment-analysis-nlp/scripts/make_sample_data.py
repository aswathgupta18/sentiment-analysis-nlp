"""Builds data/sample_reviews.csv, a small SYNTHETIC demo dataset so the repo runs out of the box.

It is template-based with deliberate noise (mixed sentiment, ~6% flipped labels), but it is
still artificial. For real numbers, train on a real dataset (see README, 'Using a real dataset').
"""
import random
from pathlib import Path

import pandas as pd

random.seed(42)

PRODUCTS = ["phone", "laptop", "headphones", "camera", "blender", "backpack", "monitor",
            "keyboard", "watch", "speaker", "vacuum cleaner", "coffee maker", "tablet", "router"]
ASPECTS = ["battery life", "build quality", "delivery", "customer support", "sound quality",
           "screen", "packaging", "price", "performance", "design", "setup"]

POS_ADJ = ["excellent", "fantastic", "amazing", "superb", "great", "impressive", "outstanding",
           "really good", "brilliant", "top notch", "very smooth", "lovely"]
NEG_ADJ = ["terrible", "awful", "disappointing", "poor", "horrible", "useless", "very bad",
           "worst", "unreliable", "frustrating", "cheaply made", "a waste of money"]
NEU_FACT = ["arrived on {day}", "comes in a {color} box", "has a {color} finish",
            "was delivered by courier", "came with a charger and a manual", "weighs about {w} grams",
            "was ordered last {day}", "is the {color} version", "needs to be plugged in for setup"]

POS_T = [
    "{a}!", "Love it, {a}.", "Would buy again, {a} {p}.", "Works great, {a}.", "So happy with this, {a}.",
    "The {p} is {a}, totally worth it.", "I love this {p}. The {asp} is {a}.",
    "{a} {p}! The {asp} exceeded my expectations.", "Very happy with the {p}, {asp} is {a}.",
    "Best purchase this year. The {p} works perfectly and the {asp} is {a}.",
    "Highly recommend this {p}, {asp} is {a}.", "The {p} is not bad at all, actually {a}.",
    "Never had any problem with this {p}. {asp} is {a}.", "Five stars. {a} {asp} on this {p}.",
    "Honestly the {p} surprised me, {asp} is {a} and it was cheap.",
]
NEG_T = [
    "{a}.", "Never buying again, {a}.", "Not worth the money, {a}.", "Stopped working, {a}.", "Regret buying this, {a}.",
    "The {p} is {a}, do not buy.", "I hate this {p}. The {asp} is {a}.",
    "{a} {p}. The {asp} was a letdown.", "Very unhappy with the {p}, {asp} is {a}.",
    "Worst purchase this year. The {p} stopped working and the {asp} is {a}.",
    "Would not recommend this {p}, {asp} is {a}.", "The {p} is not good, {asp} is {a}.",
    "Returned the {p} after two days. {asp} was {a}.", "One star. {a} {asp} on this {p}.",
    "The {p} broke within a week and support was no help, {asp} is {a}.",
]
NEU_T = [
    "It {f}.", "Received it, it {f}.", "Standard {p}, it {f}.",
    "The {p} {f}.", "I bought the {p} and it {f}.", "This {p} {f}, I have not used it much yet.",
    "The {p} {f}. Nothing else to add.", "Ordered a {p} for my brother, it {f}.",
    "The {asp} of the {p} is okay, nothing special.", "The {p} does what it says, average {asp}.",
    "It is a {p}. The {asp} is fine, not great and not bad.",
]
MIXED = [  # deliberately hard: ends up in either class
    ("The {asp} is {pa}, but the {asp2} is {na}.", None),
    ("Good {p} overall, although the {asp} is {na}.", "Positive"),
    ("The {asp} is {pa} but honestly the {asp2} ruined it, {na}.", "Negative"),
]

DAYS = ["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"]
COLORS = ["black", "white", "silver", "blue", "grey"]


def fill(t, **kw):
    return t.format(p=random.choice(PRODUCTS), asp=random.choice(ASPECTS),
                    a=kw.get("a", ""), f=kw.get("f", ""))


def noisy(s):
    r = random.random()
    if r < 0.15:
        s = s.lower()
    elif r < 0.25:
        s = s.upper() if random.random() < 0.2 else s + random.choice(["!!", "!!!", " :)", " ..."])
    if random.random() < 0.08:  # typo: swap two neighbouring letters
        i = random.randrange(1, max(2, len(s) - 2))
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s


rows = []
for _ in range(1300):
    rows.append((noisy(fill(random.choice(POS_T), a=random.choice(POS_ADJ))), "Positive"))
for _ in range(1000):
    rows.append((noisy(fill(random.choice(NEG_T), a=random.choice(NEG_ADJ))), "Negative"))
for _ in range(600):
    f = random.choice(NEU_FACT).format(day=random.choice(DAYS), color=random.choice(COLORS),
                                       w=random.randint(90, 900))
    rows.append((noisy(fill(random.choice(NEU_T), f=f)), "Neutral"))
for _ in range(300):
    t, lab = random.choice(MIXED)
    s = t.format(p=random.choice(PRODUCTS), asp=random.choice(ASPECTS), asp2=random.choice(ASPECTS),
                 pa=random.choice(POS_ADJ), na=random.choice(NEG_ADJ))
    rows.append((noisy(s), lab or random.choice(["Positive", "Negative", "Neutral"])))

# label noise, like real annotators disagreeing
labels = ["Negative", "Neutral", "Positive"]
rows = [(t, random.choice([l for l in labels if l != y]) if random.random() < 0.06 else y)
        for t, y in rows]

df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates("text").sample(frac=1, random_state=42)
out = Path(__file__).resolve().parent.parent / "data" / "sample_reviews.csv"
df.to_csv(out, index=False)
print(f"saved {len(df)} rows -> {out}")
print(df["label"].value_counts())
