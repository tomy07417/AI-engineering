"""Small labeled corpus for embedding comparisons.

Run: python src/labs/embedding_explorer/corpus.py
"""

TEXTS = [
    # Technology
    "Machine learning models require large datasets for training",
    "Neural networks are inspired by biological brain structures",
    "GPUs accelerate deep learning training significantly",
    "Transformers revolutionized natural language processing",
    # Cooking
    "Preheat the oven to 350 degrees Fahrenheit",
    "Knead the dough for about 10 minutes until smooth",
    "Season the steak with salt and pepper before grilling",
    "Let the sauce simmer on low heat for 30 minutes",
    # Sports
    "The goalkeeper made an incredible save in the final minute",
    "Marathon runners train for months before the race",
    "The tennis player served an ace to win the match",
    "Basketball requires both speed and vertical leap",
]

LABELS = ["tech"] * 4 + ["cooking"] * 4 + ["sports"] * 4


if __name__ == "__main__":
    print(f"Corpus: {len(TEXTS)} texts in {len(set(LABELS))} categories")
