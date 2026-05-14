from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

client = OpenAI(base_url="https://api.groq.com/openai/v1")
modelo = "meta-llama/llama-4-scout-17b-16e-instruct"


class ExtracaoEvento(BaseModel):
    descricao: str = Field(description="Descrição bruta do evento")
    eh_evento_calendario: bool = Field(
        description="Se esse texto descreve um evnto do calendário"
    )
    pontuacao_confianca: float = Field(description="Pontuação de confiança (0 a 1)")


class DetalhesEvento(BaseModel):
    nome: str = Field(description="Nome do evento")
    data: str = Field(
        description="Data e hora do evento. Use o formato ISO 8601 para este valor"
    )
    duracao_minutos: Optional[int] = Field(description="Duração esperada em minutos")
    participantes: list[str] = Field(description="Lista de participantes")


class ConfirmacaoEvento(BaseModel):
    mensagem_confirmacao: str = Field(
        description="Mensagem da confirmação em linguagem natural"
    )
    link_calendario: Optional[str] = Field(
        description="Justificativa para a confirmação"
    )


def extrair_informacao_evento(entrada_usuario: str) -> ExtracaoEvento:
    hoje = datetime.now()
    contexto_data = f"Hoje é {hoje.strftime('%A, %d de %B de %Y')}"

    response = client.responses.parse(
        model=modelo,
        input=f"{contexto_data} analise se o texto descreve um evento no calendário.",
        instructions=f"Extraia informações sobre um possível evento deste texto: '{entrada_usuario}' ",
        text_format=ExtracaoEvento,
    )
    return response.output_parsed


def analisar_detalhes_evento(descricao: str) -> DetalhesEvento:
    hoje = datetime.now()
    contexto_data = f"Hoje é {hoje.strftime('%A, %d de %B de %Y')}"

    response = client.responses.parse(
        model=modelo,
        input=f"{contexto_data} Extraia informações detalhadas do evento. Quando as datas fizerem referência a 'próxima terça-feira' ou datas relativas similares, use a data atual como referência.",
        instructions=f"Extraia detalhes extruturados deste texto de evento: '{descricao}' ",
        text_format=ExtracaoEvento,
    )
    return response.output_parsed


def gerar_confirmação(detalhes_evento: DetalhesEvento) -> ConfirmacaoEvento:
    response = client.responses.parse(
        model=modelo,
        input="Gere uma mensagem de confirmação natural para o evento. Assine a mensagem com seu nome: Skynet",
        instructions=f"Crie uma confirmação para este evento: '{detalhes_evento.model_dump()}' ",
        text_format=ConfirmacaoEvento,
    )
    return response.output_parsed


def processar_solicitacao_calendario(
    entrada_usuario: str,
) -> Optional[ConfirmacaoEvento]:
    extracao_inicial = extrair_informacao_evento(entrada_usuario)

    if (
        not extracao_inicial.eh_evento_calendario
        or extracao_inicial.pontuacao_confianca < 0.7
    ):
        return None

    detalhes_evento = analisar_detalhes_evento(extracao_inicial.descricao)

    return gerar_confirmação(detalhes_evento)


entrada_usuario = """Vamos fazer uma transmissão ao vivo na
próxima segunda-feira às 20h com Luiz e Felipe para apresentar
o lançamento do nosso app, deve durar cerca de 2 horas.
"""

resultado = processar_solicitacao_calendario(entrada_usuario)
if resultado:
    print(f"Confirmação: {resultado.mensagem_confirmacao}")
    if resultado.link_calendario:
        print(f"Link do calendário: {resultado.link_calendario}")
else:
    print("Isso não parece ser uma solicitação de evento de calendário.")


entrada_usuario = """Você pode enviar um e-mail para Luiz e Felipe
para discutir o roteiro do projeto?
"""

resultado = processar_solicitacao_calendario(entrada_usuario)
if resultado:
    print(f"Confirmação: {resultado.mensagem_confirmacao}")
    if resultado.link_calendario:
        print(f"Link do calendário: {resultado.link_calendario}")
else:
    print("Isso não parece ser uma solicitação de evento de calendário.")
