# TODO: Dar a ele ma vista do diretório raiz nãoé má ideia!
# --- RAG ---
# Aparenta utilizar o tools com muita frequência, ou o último
# tool call fica na stream!

# -- Setup --
from funcoes.ferramentas import *  # Feito por brevidade

# - Imports do RAG -
from langchain.agents import create_agent
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage

# Memory saver para memória de curto prazo
from langgraph.checkpoint.memory import MemorySaver

# - Criação do Agente e seus componentes -
# Houe o teste com o llama3.1:8b e com o qwen2.5:7b
# O qwen2.5 se demonstrou o mais flexível, mesmo com um número menor de parâmetros
llm = ChatOllama(model="qwen2.5:7b")
# llm = ChatOllama(model="llama3.1:8b")

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

# O prompt de sistema foi feito levando em conta os pontos fracos do BOT:
# Em geral, ele necessita de alta especficações relacionadas as suas tarefas
# Afim de dar suporte para prompts mais "livres" foi necessário adicionar
# regras mais explícitas sobre o uso de ferramentas (evitando halucinações)
sys_prompt = """
Você é um assistente virtual autônomo.
Responda APENAS em português.

REGRAS DE USO DE FERRAMENTAS (MUITO IMPORTANTE):
1. NUNCA execute múltiplas ferramentas de uma só vez (paralelamente). Execute apenas UMA ferramenta de cada vez e aguarde o resultado.
2. Não tente adivinhar o caminho ou nome de arquivos. Se você não tem certeza do nome de um documento ou dataset, use 'listar_diretorio' PRIMEIRO.
3. Se uma ferramenta retornar um erro, PARE. Não tente adivinhar formatos. Leia o erro e tente uma abordagem diferente.
4. Antes de chamar qualquer ferramenta, escreva uma breve frase explicando o que você vai fazer e por quê.
5. Se você não tiver a ferramenta adequada ao que o usuário deseja, responda que não tem a ferramenta.
6. Alguns banco de dados contém dicionários que explicam as suas variáveis, estes podem ser encontrados em 'manuais'.

Todas as perguntas do usuário serão relacionadas aos documentos que você tem acesso. Nem sempre o usuário saberá quais os documentos relevantes, cabe a você descobrir quais utilizar.
"""

ferramentas = [calcular_churn_rate]
ferramentas_experimentais = [listar_diretorio,
               indexar_documento,
               buscar_docs,
               abrir_doc,
               indexar_dataset,
               dataset_info,
               dataset_query]
ferramentas += ferramentas_experimentais
middleware = [trimming]

agente = create_agent(
    llm,
    tools=ferramentas,
    system_prompt=sys_prompt,
    checkpointer=MemorySaver(),
    middleware=middleware
)

# -- Funções --
# - Funções de conversa -
# Aqui se encontra as funções para acessar o RAG em si

# OBSOLETO - Função de invocação direta
# def conv_invoke(prompt: str = "", sys_info = False) -> str:
#     """Visualiza resposta do agente após finalizada."""
#     resp = agente.invoke({"messages": [{"role": "user", "content": prompt}]},
#                        {"configurable": {"thread_id": "1", "recursion_limit": 5}})
#     if sys_info:
#         print("---\n")
#         for m in resp['messages']:
#             m.pretty_print()
#         print("---\n")
#     return resp['messages'][-1].content

# OBSOLETO - função de invocação com stream
# def conv_stream(prompt: str = "", sys_info: bool = False) -> str:
#     """Visualiza resposta do agente a medida que ela é formada."""
#     stream = agente.stream(
#         {"messages": [{"role": "user", "content": prompt}]},
#         {
#             "configurable": {
#                 "thread_id": "1",
#                 "recursion_limit": 8
#             }
#         },
#         stream_mode=["updates"]   # eventos do agente (tool calls, etc.)
#     )

#     final_message = None

#     for chunk in stream:

#         if sys_info:
#             print(chunk)
#             print()

#         # captura mensagens quando aparecem
#         if isinstance(chunk, dict):
#             for node_output in chunk.values():
#                 if isinstance(node_output, dict) and "messages" in node_output:
#                     msgs = node_output["messages"]
#                     if msgs:
#                         final_message = msgs[-1]

#     if final_message:
#         return final_message.content

#     return ""


def chat_rag():

    print("Chat iniciado. Digite 'sair' para sair.\n")

    while True:

        prompt = input("Usuário: ")

        if prompt.lower() in ['sair', 's']:
            print("Encerrando.")
            break

        print("RAG:", end=" ", flush=True)

        stream = agente.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            {
                "configurable": {
                    "thread_id": "1"
                },
                "recursion_limit": 25
            },
            stream_mode=["updates", "messages"]
        )

        for mode, chunk in stream:
            if mode == "messages":

                token, metadata = chunk

                if token.content:
                    print(token.content, end="", flush=True)


            elif mode == "updates":  # Tools e relacionados

                for node, data in chunk.items():

                    if not data or "messages" not in data:
                        continue

                    for msg in data["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:

                            for call in msg.tool_calls:
                                print(
                                    f"\n\n[Chamada da Tool] {call['name']}({call['args']})\n",
                                    flush=True
                                )
                        if msg.__class__.__name__ == "ToolMessage":
                            print(
                                f"\n[Resultado da Tool] {msg.content}\n",
                                flush=True
                            )

        print("\n")
