import tokenization


raw_text = 'aa <e1> ddd </e1>  aa $ # '

tokenizer = tokenization.FullTokenizer('model/wwm_uncased_L-24_H-1024_A-16/vocab.txt')
tokens = tokenizer.tokenize(raw_text)

print(tokens)


