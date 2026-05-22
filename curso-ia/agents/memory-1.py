from warnings import filters
from mem0 import MemoryClient
from dotenv import load_dotenv

load_dotenv()

client = MemoryClient()

# messages = [
#     {
#         "role": "user",
#         "content": "Meu nome é Luiz Felipe e eu gosto de fazer automações com IA!",
#     },
#     {
#         "role": "assistant",
#         "content": "Oi Luiz Felipe! Anotei que você gosta de contruir automações com IA! Vou manter isso em mente para recomendações e discussões relacionadas.",
#     },
# ]

# client.add(messages, user_id="luiz felipe")

client.add("Sou o Luiz Felipe e gosto de robótica.", user_id="luiz felipe")

query = "Qual o meu nome"
response = client.search(query, filters={"user_id": "luiz felipe"})
response["results"][0]["memory"]
