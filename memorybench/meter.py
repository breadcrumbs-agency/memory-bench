"""The meter: one token count and one cut rule for every system."""

RULE = ("tokens(item) = ceil(characters / 4); the items stay in rank order; "
        "the first item that does not fit ends the context")


def tokens(text):
    return (len(text) + 3) // 4


def cut(items, budget):
    """(the items that fit in budget, the tokens that they use). An item that is not text is an error."""
    out, used = [], 0
    for item in items:
        if not isinstance(item, str):
            raise TypeError("recall() must return text items, not %s" % type(item).__name__)
        n = tokens(item)
        if used + n > budget:
            break
        out.append(item)
        used += n
    return out, used


def norm(text):
    """Lower case and one space between words: the form in which evidence is matched."""
    return " ".join(text.lower().split())
