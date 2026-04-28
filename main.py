# TODO: Implementar histórico de conversas (arquivo)
# TODO: Arquivo backlog com bot abertos (para fechar eles caso tenham sido abertos em outra sessão?)
# Loop principal do projeto
import subprocess
import time
import requests
import sys
import os

# - Constantes -
DB_MODE = True

# -- Main --
import funcoes.rag as rag
from funcoes.ferramentas import print_sys_msg

# -- Flask
from flask import Flask, request, Response, send_from_directory

# -- subprocess do Ollama --
ps = None

# Diretório playground onde os arquivos enviados pela interface são salvos
PLAYGROUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'playground')

def ollama_rodando():
    try:
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
    if ps:
        ps.terminate()
        ps.wait()


# -- Interface visual (Flask) --
app = Flask(__name__, static_folder='interface', static_url_path='/static')

@app.route('/')
def index():
    return send_from_directory('interface', 'index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Recebe um arquivo e salva no playground para uso do agente."""
    if 'arquivo' not in request.files:
        return {'erro': 'Nenhum arquivo enviado.'}, 400
    arquivo = request.files['arquivo']
    if not arquivo.filename:
        return {'erro': 'Nome de arquivo vazio.'}, 400
    os.makedirs(PLAYGROUND_DIR, exist_ok=True)
    destino = os.path.join(PLAYGROUND_DIR, arquivo.filename)
    arquivo.save(destino)
    return {'ok': True, 'arquivo': arquivo.filename}

@app.route('/chat', methods=['POST'])
def chat():
    dados = request.get_json()
    mensagem = dados.get('mensagem', '')

    def gerar():
        for chunk in rag.chat_rag(mensagem):
            if chunk:
                yield chunk

    return Response(gerar(), mimetype='text/plain; charset=utf-8')


if __name__ == '__main__':
    abrir_ollama()

    # Seleção de modelo antes de iniciar funciona tanto no modo web quanto CLI
    print("\n=== PI-RAG · Selecione o Modelo ===\n")
    modelo_op = rag.conj_menu_cli(
        ops=[f"Modelo {m}" for m in rag.modelos],
        escolha=-1,
        clear_cli=False   # mantém mensagens de inicialização visíveis
    )

    if modelo_op < 0:
        print_sys_msg("Encerrando...")
        fechar_ollama()
        sys.exit(0)

    # Inicializa o RAG com o modelo escolhido e todas as ferramentas sempre ativas
    rag.configurar_rag(modelo_op=modelo_op, modo=1)

    if '--cli' in sys.argv:
        # Modo terminal
        rag.chat_rag_cli()
        fechar_ollama()
    else:
        # Modo interface web
        print(f"\nAcesse em: http://localhost:8000\n")
        app.run(host='0.0.0.0', port=8000, debug=False)