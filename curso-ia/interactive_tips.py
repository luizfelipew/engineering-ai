# Dicas para usar o modo interativo no VS Code com Python

# 1. Para exibir variáveis automaticamente no modo interativo:
# Coloque apenas o nome da variável na última linha da célula
my_variable = [1, 2, 3, 4, 5]
my_variable  # Esta linha fará com que o conteúdo seja exibido

# 2. Use print() para informações adicionais:
print(f"Tipo da variável: {type(my_variable)}")
print(f"Tamanho: {len(my_variable)}")

# 3. Para estruturas de dados complexas, use pprint:
from pprint import pprint

complex_data = {
    "tokens": ["Machine", "Learning", "é", "um", "campo"],
    "count": 5,
    "metadata": {"language": "pt", "encoding": "utf-8"},
}
pprint(complex_data)

# 4. Para DataFrames (se usando pandas):
# import pandas as pd
# df = pd.DataFrame(data)
# df  # Exibe a tabela formatada

# 5. Para visualizações:
# import matplotlib.pyplot as plt
# plt.plot([1, 2, 3], [4, 5, 6])
# plt.show()  # Necessário para mostrar o gráfico

# 6. Configurações úteis para o VS Code:
# - Certifique-se de que o Python Interactive Window está habilitado
# - Use Ctrl+Shift+P e procure por "Python: Start REPL" se necessário
# - Configure o interpretador Python correto (Ctrl+Shift+P > "Python: Select Interpreter")

# Exemplo prático com tokenização:
import nltk

nltk.download("punkt_tab")

text = "Este é um exemplo de texto para tokenização."
tokens = nltk.word_tokenize(text)

# Mostrar informações sobre os tokens
print("Texto original:", text)
print("Tokens:", tokens)
print("Número de tokens:", len(tokens))

# Exibir a lista de tokens (será mostrada automaticamente no modo interativo)
tokens


