__all__ = ['atomwise_tokenizer', 'kmer_tokenizer', 'tokens_to_mer']



def atomwise_tokenizer(smi, exclusive_tokens = None):
    """
    test command:
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens

    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return tokens

# Cell

def kmer_tokenizer(smiles, ngram=4, stride=1, remove_last = False, exclusive_tokens = None):
    units = atomwise_tokenizer(smiles, exclusive_tokens = exclusive_tokens) #collect all the atom-wise tokens from the SMILES
    if ngram == 1:
        tokens = units
    else:
        tokens = [tokens_to_mer(units[i:i+ngram]) for i in range(0, len(units), stride) if len(units[i:i+ngram]) == ngram]

    if remove_last:
        if len(tokens[-1]) < ngram: #truncate last whole k-mer if the length of the last k-mers is less than ngram.
            tokens = tokens[:-1]
    return tokens

def tokens_to_mer(toks):
    return ''.join(toks)
