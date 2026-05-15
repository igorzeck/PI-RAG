# TODO: Função para lidar com saídas inesperadas
# TODO: Consertar bug em que o bot para no meio do fluxo de geração de texto.
# TODO: Churn rate retorna clientes como 1 para chance de Churn!
# TODO: Pro Churn não só retornar 1 e 0, retornar também probabilidade
# Loop principal do projeto
# Para abrir o navegador na interface visual
import webbrowser
from threading import Timer

import pathlib
import subprocess
import time
import requests
import sys
import os
import atexit
import signal

# - Constantes -
DB_MODE = True
TMP_PREF = 'recursos/'
TMP_SUFF = '.tmp'
# File containing current session info
OLLAMA_PS_PATH = pathlib.Path(TMP_PREF + '.ollama_ps' + TMP_SUFF)
LINK_INTERFACE = 'http://localhost:8000'
# -- Main --
import funcoes.rag as rag
from funcoes.ferramentas import print_sys_msg

# -- Flask
from flask import Flask, request, Response, send_from_directory

# -- subprocess do Ollama --
ps = None

# Diretório playground onde os arquivos enviados pela interface são salvos
PLAYGROUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'playground')

def abrir_navegador():
    webbrowser.open_new(LINK_INTERFACE)

def ollama_rodando():
    try:
        requests.get("http://localhost:11434/api/tags")
        return True
    except requests.exceptions.ConnectionError:
        return False


def abrir_ollama():
    """Abre ollama caso já não tenha sido aberto pelo programa em outra sessão."""
    global ps

    # Primeiro checa se o Ollama foi aberto em alguma sessão anterior
    if OLLAMA_PS_PATH.is_file():
        # Se foi aberto manda sinal para "matar" o processo e recria ele
       print_sys_msg("Arquivo de sessão anterior achado!")
       with open(OLLAMA_PS_PATH, 'r') as arq:
            _pid_raw = arq.readline()
            if _pid_raw.isnumeric():
                _pid = int(_pid_raw)
                try:
                    os.kill(_pid, signal.SIGTERM)
                except ProcessLookupError:
                    print_sys_msg("Processo Ollama anterior não achado. Assume-se fechado!", type="Aviso")
                    # Deleta arquivo temporário

    if not ollama_rodando():
        try:
            ps = subprocess.Popen(['ollama', 'serve'],
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            print_sys_msg("Inicializando o Ollama...")
            time.sleep(5)

            # Cria arquivo temporário com o processo aberto
            with open(OLLAMA_PS_PATH, "w") as arq:
                arq.write(str(ps.pid))
        except Exception as e:
            print_sys_msg(f"Ollama não incializou: {e}", type="Erro")


def fechar_ollama():
    global ps

    if ps:
        print_sys_msg("Fechando o Ollama...")
        ps.terminate()
        ps.wait()
        ps = None
        
        # Deleta arquivo contendo PID do processo (caso exista)
        if OLLAMA_PS_PATH.is_file():
            os.remove(OLLAMA_PS_PATH)


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

def cleanup():
    """Função executada nas saídas do código"""
    fechar_ollama()

atexit.register(cleanup)

if __name__ == '__main__':
    abrir_ollama()

    # Seleção de modelo antes de iniciar funciona tanto no modo web quanto CLI
    print("\n=== PI-RAG · Selecione o Modelo ===\n")
    modelo_op = rag.conj_menu_cli(
        ops=[f"Modelo {m}" for m in rag.modelos],
        escolha=-1,
        clear_cli=False,   # mantém mensagens de inicialização visíveis
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
        print(f"\nAcesse em: {LINK_INTERFACE}\n")

        # Abre o navegador padrão
        if not os.environ.get("WERKZEUG_RUN_MAIN"):
            Timer(1, abrir_navegador).start()
        # Talvez necessite dar um refresh ná página se aberto assim...
        app.run(host='0.0.0.0', port=8000, debug=False)