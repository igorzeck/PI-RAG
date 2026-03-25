# TODO: Implementar histórico de conversas (arquivo)
# Loop principal do projeto
# - Constantes -
DB_MODE = True
# -- Main --
import funcoes.rag as rag

def main():
    rag.configurar_rag(0, 0)
    rag.chat_rag_cli()

if __name__ == '__main__':
    main()