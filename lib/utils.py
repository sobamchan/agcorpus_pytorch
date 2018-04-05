

def preprocess(sent):
    sent = sent.replace(', ', ' , ')
    sent = sent.replace('. ', ' . ')
    sent = sent.replace('( ', ' ( ')
    sent = sent.replace(') ', ' ) ')
    sent = sent.replace("' ", " ' ")
    return sent


def pad_to_n(words, n, pad_idx):
    words_n = len(words)
    if words_n > n:
        return words[:n]
    elif words_n == n:
        return words
    else:
        words += [pad_idx] * (n - words_n)
        return words
