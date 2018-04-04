

def preprocess(sent):
    sent = sent.replace(', ', ' , ')
    sent = sent.replace('. ', ' . ')
    sent = sent.replace('( ', ' ( ')
    sent = sent.replace(') ', ' ) ')
    sent = sent.replace("' ", " ' ")
    return sent


def pad_to_n(batch_words, n, pad_idx):
    new_batch = []
    for words in batch_words:
        words_n = len(words)
        if words_n > n:
            new_batch.append(words[:n])
        elif words_n == n:
            new_batch.append(words)
        else:
            words += [pad_idx] * (n - words_n)
            new_batch.append(words)

    return new_batch
