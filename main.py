# TODO: Implementar histórico de conversas (arquivo)
# TODO: Arquivo backlog com bot abertos (para fechar eles caso tenham sido abertos em outra sessão?)
# Loop principal do projeto
import subprocess
import time
import requests
# - Constantes -
DB_MODE = True
# -- Main --
import funcoes.rag as rag

# -- subprocess do Ollama --
# TODO: Mover essas funções do Ollama para outro lugar
ps = None

def ollama_rodando():
    try:
        # Check the default Ollama port
        requests.get("http://localhost:11434/api/tags")
        return True
    except requests.exceptions.ConnectionError:
        return False


def abrir_ollama():
    global ps

    if not ollama_rodando():
        try:
            ps = subprocess.Popen(['ollama', 'serve'], 
                                    stdout=subprocess.DEVNULL, 
                                    stderr=subprocess.DEVNULL)
            print("Inicializando o Ollama...")
            time.sleep(5)
        except Exception as e:
            print(f"Ollama não incializou: {e}")


def fechar_ollama():
    # Fechar apenas se o subprocesso
    # for aberto aqui por esse script
    if ps:
        ps.terminate()
        ps.wait()

def main():
    # Abre o Ollama
    abrir_ollama()

    # Configura o RAG
    # TODO: Interface visual para configurar o RAG!
    rag.configurar_rag(0, 0)
    
    # CLI
    rag.chat_rag()

    # Fecha o Ollama
    fechar_ollama()

if __name__ == '__main__':
    main()