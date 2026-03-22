# TODO: Dar a ele ma vista do diretório raiz nãoé má ideia!
# --- RAG ---
# Aparenta utilizar o tools com muita frequência, ou o último
# tool call fica na stream!

# region: Setup --
import os
from funcoes.ferramentas import *  # Feito por brevidade
from colorama import Fore

# - Imports do RAG -
from langchain.agents import create_agent
# from typing import TypedDict, Sequence
# from langchain_core.messages import BaseMessage

# Memory saver para memória de curto prazo
from langgraph.checkpoint.memory import MemorySaver

# - Criação do Agente e seus componentes -
# Houe o teste com o llama3.1:8b e com o qwen2.5:7b
# O qwen2.5 se demonstrou o mais flexível, mesmo com um número menor de parâmetros
# Devido ao reasoning interno do bot, o num_ctx foi aumenta por volta de 15X

llm = ChatOllama(model="qwen2.5:7b", num_ctx=32768)
# llm = ChatOllama(model="llama3.1:8b")

# class AgentState(TypedDict):
#     messages: Sequence[BaseMessage]

# region: Prompt de sistema -
# O prompt de sistema foi feito levando em conta os pontos fracos do BOT:
# Em geral, ele necessita de alta especficações relacionadas as suas tarefas
# Afim de dar suporte para prompts mais "livres" foi necessário adicionar
# regras mais explícitas sobre o uso de ferramentas (evitando halucinações)
sys_prompt_experimental = """
Você é um assistente virtual autônomo.
Responda APENAS em português.

REGRAS DE USO DE FERRAMENTAS (MUITO IMPORTANTE):
1. NUNCA execute múltiplas ferramentas de uma só vez (paralelamente). Execute apenas UMA ferramenta de cada vez e aguarde o resultado.
2. Não tente adivinhar o caminho ou nome de arquivos. Se você não tem certeza do nome de um documento ou dataset, use 'listar_diretorio' PRIMEIRO.
3. Se uma ferramenta retornar um erro, PARE. Não tente adivinhar formatos. Leia o erro e tente uma abordagem diferente.
4. Antes de chamar qualquer ferramenta, escreva uma breve frase explicando o que você vai fazer e por quê.
5. Se você não tiver a ferramenta adequada ao que o usuário deseja, responda que não tem a ferramenta.
6. Antes de olhar um banco de dados ou dataset, verifique se contém dicionários que explicam as suas variáveis, estes podem ser encontrados em 'manuais'.

Todas as perguntas do usuário serão relacionadas aos documentos que você tem acesso. Nem sempre o usuário saberá quais os documentos relevantes, cabe a você descobrir quais utilizar.
"""

sys_prompt_padrao = """
Você é um assistente virtual autônomo.
Responda APENAS em português.
Vocẽ funciona em dois estágios: PENSAMENTO e FALA. Inicie E termine seções de pensamento com '[Pensamento]'

REGRAS DE USO DE FERRAMENTAS (MUITO IMPORTANTE):
1. NUNCA execute múltiplas ferramentas de uma só vez (paralelamente). Execute apenas UMA ferramenta de cada vez e aguarde o resultado.
2. Se uma ferramenta retornar um erro, PARE. Não tente adivinhar formatos. Leia o erro e tente uma abordagem diferente.
3. PENSAMENTO: Antes de chamar qualquer ferramenta, escreva uma breve frase explicando o que você vai fazer e por quê.
4. Assuma que as perguntas podem ser respondidas utilizando as ferramentas relevantes, e que SE dados forem necessários são SEMPRE parâmetros das ferramentas.
5. Se você não tiver a ferramenta adequada ao que o usuário deseja, responda que não tem a ferramenta.
6. Sempre assuma não saber a resposta para a pergunta do usuário MESMO PARA COISAS BÁSICAS, e DEPENDA APENAS DO RESULTADO das ferramentas e das informações delas.
7. Seja breve com respostas que não podem ser feitas com uso de uma ferramenta. Sempre conidere utilizar ferramentas ANTES de responder.
8. Apenas responda o que o usuário perguntou, NADA MAIS que o que foi perguntado.
"""
# endregion: Prompt de sistema -

ferramentas_padrao = [calcular_churn_rate]
# As experimentais olham os arquivos no diretório de trabalho
ferramentas_experimentais = [listar_diretorio,
               indexar_documento,
               buscar_docs,
               abrir_doc,
               dataset_info,
               dataset_query]

middleware = [trimming]

modo_chat = ""
agente = None

# endregion

# region: Funções --
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
def configurar_rag(auto_inicializar = False):
    """
    Inicializa o RAG com configurações padrões;
    params:
    modo_chat: "Experimental" (e) ou "Padrão" (p)
    """
    global agente
    global modo_chat
    
    if auto_inicializar:
        input_usuario = "1"
    else:
        print("Escolha o modo do RAG:\n1. Padrão - Capaz de chamar funções de calculo.\n2. Experimental - Capaz de acessar documentos.\n")
        input_usuario = input(": ")

    if input_usuario == "1":
        agente = create_agent(
            llm,
            tools=ferramentas_padrao,
            system_prompt=sys_prompt_padrao,
            checkpointer=MemorySaver(),
            middleware=middleware
        )
        modo_chat = "Padrão"
    elif input_usuario == "2":
        agente = create_agent(
            llm,
            tools=ferramentas_experimentais,
            system_prompt=sys_prompt_experimental,
            checkpointer=MemorySaver(),
            middleware=middleware
        )
        modo_chat = "Experimental"
    else:
        print("Opção inválida!")
        exit(1)


def chat_rag(modo_db: bool = False):
    if not agente:
        print("Agente não inicializado!")
        exit(1)
    # Limpa o terminal antes de iniciar o chat
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"Chat iniciado. Digite 'sair' para sair.\nModo: {modo_chat}", end="\n\n")

    # TODO: Deixar como middleware para poder pasar pelo trimming!
    printed_msg_ids = set()

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

                # Imprime apenas o stream do RAG, ignorando ToolMessages para não duplicar
                if getattr(token, "type", "") == "AIMessageChunk" and token.content:
                    print(token.content, end="", flush=True)

            elif mode == "updates":  # Tools e relacionados

                for node, data in chunk.items():
                    # Mostra apenas mensagens
                    if (not data) or ("messages" not in data):
                        continue

                    print(Fore.LIGHTBLACK_EX, end="")

                    for msg in data.get("messages", []):
                        # Pula a mensagem se ela já foi processada antes
                        if getattr(msg, "id", None) in printed_msg_ids:
                            continue
                        # Adiciona mensagem para não ser printada de novo
                        if hasattr(msg, "id") and msg.id:
                            printed_msg_ids.add(msg.id)
                        
                        if hasattr(msg, "tool_calls") and msg.tool_calls:

                            for call in msg.tool_calls:
                                if modo_db:
                                    print(
                                        f"\n[Chamada da Tool] {call['name']}({call['args']})\n",
                                        flush=True
                                    )
                                else:
                                    print(
                                        f"\n{Fore.LIGHTGREEN_EX}Chamada da Tool{Fore.RESET}: {call['name']}({call['args']})\n",
                                        flush=True
                                    )
                        if msg.__class__.__name__ == "ToolMessage" and modo_db:
                            print(
                                f"\n[Resultado da Tool] {msg.content}\n",
                                flush=True
                            )

                    
                    print(Fore.RESET, end="")

        print("\n")
# endregion