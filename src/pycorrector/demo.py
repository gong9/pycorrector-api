from pycorrector.macbert.macbert_corrector import MacBertCorrector

m = MacBertCorrector("shibing624/macbert4csc-base-chinese")

i = m.correct("今天新情很好")
print(i)
