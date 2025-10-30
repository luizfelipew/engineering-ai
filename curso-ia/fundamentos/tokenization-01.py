import nltk

# Baixar os dados necessários do NLTK
nltk.download("punkt_tab")

# Definir o texto para tokenização
text = "Machine Learning é um campo da inteligência Artificial que permite que computadores aprendam padrões a partir de dados. Sem serem programados explicitamente para cada tarefa."

word_tokens = nltk.word_tokenize(text)
print(word_tokens)

sentences_tokens = nltk.sent_tokenize(text)
print(sentences_tokens)


def preprocess(text):
    # Tokenização em palavras
    tokens = nltk.word_tokenize(text.lower())

    return [word for word in tokens if word.isalnum()]


documents = [
    "Machine learning é o aprendizado automático de máquinas a partir de dados.",
    "Ele permite que sistemas façam previsões e decisões sem programação explícita.",
    "É usado em áreas como reconhecimento de voz, imagens e recomendação de conteúdo.",
]

preprocessed_docs = [" ".join(preprocess(doc)) for doc in documents]
print(preprocessed_docs)
