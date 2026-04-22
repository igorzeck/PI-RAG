# --- RAG ---
# Aparenta utilizar o tools com muita frequência, ou o último
# tool call fica na stream!
# NOTE: Possível parar um modelo específico com: ollama.stop('llama3') 

# region: Setup --
import os
from funcoes.ferramentas import *  # Feito por brevidade

# - Imports do RAG -
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
# from typing import TypedDict, Sequence
# from langchain_core.messages import BaseMessage

# Memory saver para memória de curto prazo
from langgraph.checkpoint.memory import MemorySaver

# - Criação do Agente e seus componentes -
# Houe o teste com o llama3.1:8b e com o qwen2.5:7b
# O qwen2.5 se demonstrou o mais flexível, mesmo com um número menor de parâmetros
# Devido ao reasoning interno do bot, o num_ctx foi aumenta por volta de 15X
# region: Configurações do RAG
modelos = [
        "qwen2.5:3b",
        "qwen2.5:7b",
        "qwen3:4b",
        "qwen3.5:2b",
        "qwen3.5:4b",
        "llama3.2:3b",
    ]

# Pelo visto é necessário saber com antecedência!
suporte_thinking = [
    False,
    False,
    True,
    True,
    False,
]

thinker = False
# endregion

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
6. Antes de indexar um arquivo, verifique se não há outros arquivos mais adequados para indexar.
7. Todas as perguntas do usuário serão relacionadas aos documentos que você tem acesso. Nem sempre o usuário saberá quais os documentos relevantes, cabe a você descobrir quais utilizar.
"""
# 8. Caso o usuário fale de algo claramente não relacionado aos documentos ou ao trabalho, peça pra ele explicar a relevância do assunto, e se ele continuar, então, finalize IMEDIATAMENTE a conversa utilizando a ferramenta 'finalizar_conversa'
# """

sys_prompt_padrao = """
Você é um assistente virtual autônomo.
Responda APENAS em português.

REGRAS DE USO DE FERRAMENTAS (MUITO IMPORTANTE):
1. NUNCA execute múltiplas ferramentas de uma só vez (paralelamente). Execute apenas UMA ferramenta de cada vez e aguarde o resultado.
2. Se uma ferramenta retornar um erro, PARE. Não tente adivinhar formatos. Leia o erro e tente uma abordagem diferente.
3. Antes de chamar qualquer ferramenta, escreva uma breve frase explicando o que você vai fazer e por quê.
4. Assuma que as perguntas podem ser respondidas utilizando as ferramentas relevantes, e que SE dados forem necessários são SEMPRE parâmetros das ferramentas.
5. Se você não tiver a ferramenta adequada ao que o usuário deseja, responda que não tem a ferramenta.
6. Sempre assuma não saber a resposta para a pergunta do usuário MESMO PARA COISAS BÁSICAS, e DEPENDA APENAS DO RESULTADO das ferramentas e das informações delas.
7. Seja breve com respostas que não podem ser feitas com uso de uma ferramenta. Sempre conidere utilizar ferramentas ANTES de responder.
8. Apenas responda o que o usuário perguntou, NADA MAIS que o que foi perguntado.
"""
# Caso o usuário fale de algo claramente não relacionado aos documentos ou ao trabalho, peça pra ele explicar a relevância do assunto, então finalize IMEDIATAMENTE a conversa utilizando a ferramenta 'finalizar_conversa'
# """
# endregion: Prompt de sistema -

ferramentas_base = [finalizar_conversa]

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
# - Auxiliares -
# TODO: Talvez enter para opção padrão?
def conj_menu_cli(ops: list[str], escolha: int = -1, sair_como_ultima = False, clear_cli = True) -> int:
    """
    Menu com lista de opções que retorna a escolhida
    params:
        escolha - Se um número menor que 0, conjura menu de opções.
        sair_como_ultima - Se True, adiciona uma opção para saída.
        clear - Se True, limpa o terminal.
    return:
        Retorna id da opção escolhida (assim como na lista).
    """
    if escolha >= 0:
        return escolha
    
    # Não recomendado deixar como True (apaga todo histório do terimal...)
    if clear_cli:
        os.system('cls' if os.name == 'nt' else 'clear')

    saidas = ["sair", "s", "quit", "q", "exit"]

    if sair_como_ultima:
        saidas.append(len(ops))

    while True:
        print("Escolha uma das opções. Ou escreva 's' para sair.")
        
        for i, op in enumerate(ops):
            print(f"{i}. {op}")
        
        input_usuario = input("\n: ")
        if input_usuario.lower() in saidas:
            print_sys_msg("Saindo...")
            return -1
        
        if not input_usuario.isnumeric():
            print_sys_msg("Opção inválida! Necessário número!", "Erro")
            continue
        
        if int(input_usuario) > len(ops):
            print_sys_msg("Escolha uma das opções!", "Erro")
            continue
        
        return int(input_usuario)


# - Configurações -
def configurar_rag(modelo_op: int = -1, modo: int = -1, with_debug_output = False):
    """
    Inicializa o RAG com configurações padrões;
    params:
        modelo: int - Modelo (índice lista modelos)
        modo: int - Modo Padrão (0) e Experimental (1)
    """
    global agente
    global modo_chat
    global modo_db
    global thinker

    modo_db = with_debug_output

    # Menu 1 - Modelo do Agente
    op_= conj_menu_cli(ops=[f"Modelo {Fore.LIGHTCYAN_EX}{modelo.capitalize()}{Fore.RESET}." for modelo in modelos],
                       escolha=modelo_op)

    if op_ < 0:
        print_sys_msg("Encerrando...")
        exit(0)
    
    modelo = modelos[op_]
    thinker = suporte_thinking[op_]

    llm = ChatOllama(model=modelo,
                     num_ctx=32768,
                     reasoning=thinker)  # Nem todos modelos tem suporte!

    # Menu 2 - Modo do Agente
    op_ = conj_menu_cli(ops=[
        f"{Fore.LIGHTCYAN_EX}Padrão{Fore.RESET} - Capaz de chamar funções de calculo.",
        f"{Fore.LIGHTCYAN_EX}Experimental{Fore.RESET} - Capaz de acessar documentos.",
    ], escolha=modo)

    if op_ < 0:
        print_sys_msg("Encerrando...")
        exit(0)
    
    if op_ == 0:
        agente = create_agent(
            llm,
            tools=ferramentas_base + ferramentas_padrao,
            system_prompt=sys_prompt_padrao,
            checkpointer=MemorySaver(),
            middleware=middleware
        )
        modo_chat = "Padrão"
        return
    elif op_ == 1:
        agente = create_agent(
            llm,
            tools=ferramentas_base + ferramentas_experimentais,
            system_prompt=sys_prompt_experimental,
            checkpointer=MemorySaver(),
            middleware=middleware
        )
        modo_chat = "Experimental"
        return

# - Funções de conversa -
def chat_rag(prompt: str):
    """
    Pega últimas mensagens do chat (desde de a última chamada da função)
    Essa função é essencialmente um wrapper da função 'stream'
    """
    if not agente:
        print_sys_msg("Agente não inicializado!", "Erro")
        return "Agente não inicializado!"

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

    printed_msg_ids = set()
    em_reasoning = False

    for mode, chunk in stream:
        gerado = ""
        # TODO: Deixar esse if mais limpo
        if mode == "messages" and (not modo_processamento or modo_db):
            token, metadata = chunk
            additional_kwargs = getattr(token, "additional_kwargs", "")
            # Imprime apenas o stream do RAG, ignorando ToolMessages para não duplicar
            if getattr(token, "type", "") == "AIMessageChunk" and token.content:
                gerado += token.content
                if em_reasoning:
                    em_reasoning = False
            elif "reasoning_content" in additional_kwargs:
                # Se o modelo não suportar thiking isso vai ser redundante
                if modo_db:
                    gerado += additional_kwargs["reasoning_content"]
                elif not em_reasoning:
                    em_reasoning = True
                    gerado += "Pensando..."

        elif mode == "updates":  # Tools e relacionados

            for node, data in chunk.items():
                # Mostra apenas mensagens
                if (not data) or ("messages" not in data):
                    continue

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
                                gerado += f"\n[Chamada da Tool] {call['name']}({call['args']})\n"
                            else:
                                gerado += f"\nChamada da Tool: {call['name']}({call['args']})\n"
                    if msg.__class__.__name__ == "ToolMessage" and modo_db:
                        gerado += f"\n[Resultado da Tool] {msg.content}\n"
        yield gerado

def chat_rag_cli():
    """Função para conversa em modo CLI. Utilizada para testes."""
    os.system('cls' if os.name == 'nt' else 'clear')

    if not agente:
        print_sys_msg("Agente não inicializado!", "Erro")
        exit(1)
    
    print(f"Chat iniciado. Digite 'sair' para sair.\nModo: {modo_chat}", end="\n\n")

    # NOTE: Não precisa passar pelo trimming, já que cada um é único
    printed_msg_ids = set()
    em_reasoning = False

    while True:
        prompt = input(f"{Fore.BLUE}Usuário{Fore.RESET}: ")

        # Saída
        if prompt.lower() in ['sair', 's']:
            print_sys_msg("Encerrando...")
            break

        print(f"\n{Fore.LIGHTBLUE_EX}RAG{Fore.RESET}:", end=" ", flush=True)

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
            # TODO: Deixar esse if mais limpo
            if mode == "messages" and (not modo_processamento or modo_db):
                token, metadata = chunk
                additional_kwargs = getattr(token, "additional_kwargs", "")
                # Imprime apenas o stream do RAG, ignorando ToolMessages para não duplicar
                if getattr(token, "type", "") == "AIMessageChunk" and token.content:
                    print(token.content, end="", flush=True)
                    if em_reasoning:
                        em_reasoning = False
                elif "reasoning_content" in additional_kwargs:
                    # Se o modelo não suportar thiking isso vai ser redundante
                    if modo_db:
                        print_sys_msg(additional_kwargs["reasoning_content"], end="", flush=True)
                    elif not em_reasoning:
                        em_reasoning = True
                        print_sys_msg("Pensando...", end="\n", flush=True)

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