from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = OpenAI(base_url="https://api.groq.com/openai/v1")


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


response = client.responses.parse(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    input="Luiz Felipe e Cícero vão gravar uma aula na terça-feira",
    instructions="Extraia informações do evento.",
    text_format=CalendarEvent,
)

event = response.output_parsed
print(event)

print(event.participants)

print(event.model_dump_json(indent=2))

# Essa é a forma que não responde com a formatção do mesmo json

response2 = client.responses.create(
    model="llama-3.1-8b-instant",
    input="""Extraia informações do evento: Luiz Felipe e Cícero vão gravar uma aula na terça-feira
EXMEPLO DA FORMATAÇÃO
    { 
        "formato_saída": {
            "pessoas": ["..."],
            "acao": "...",
            "tipo_evento": "...",
            "data": "..."
        }
    }
""",
)

print(response2.output_text)
