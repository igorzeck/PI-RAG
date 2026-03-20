# -- Main --
# Contém o loop principal do projeto
import funcoes.rag as rag

def main():
    # Loop principal
    # Por agora só chama a função de chat continuamente
    prompt = ""
    while prompt != 'sair':
        resposta = rag.conv(prompt)
        print(">", resposta)
        prompt = input("> ")


if __name__ == '__main__':
    main()