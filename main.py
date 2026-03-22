# - Constantes -
DB_MODE = True
# -- Main --
# TODO: Implementar histórico de conversas
# Contém o loop principal do projeto
import funcoes.rag as rag

def main():
    rag.chat_rag()

# def main():
#     # Loop principal
#     # Por agora só chama a função de chat continuamente
#     prompt = open('.testes/p.txt').read()
#     if prompt:
#         print("Usuário:", prompt)
#     while prompt.lower() != 'sair':
#         if rag.e_valido(prompt):
#             resposta = rag.conv_stream(prompt, DB_MODE)
#             print("Agente:", resposta)
#         prompt = input("Usuário: ")


if __name__ == '__main__':
    main()