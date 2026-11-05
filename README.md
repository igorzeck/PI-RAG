# README

Este arquivo apresenta informações básicas acerca do projeto ["PI-RAG"](https://github.com/igorzeck/PI-RAG/tree/main)

<img width="645" height="499" alt="image" src="https://github.com/user-attachments/assets/dca8d9a6-6b8f-4e79-ba6e-a87cc22f9c84" />

Imagem: Interface visual do *bot*.

## Organização do projeto

O projeto é organizado da seguinte maneira:

- No diretório **funcoes** encontram-se as funções e o RAG criado para este projeto;
- No diretório **sandbox** encontram-se o playground acessível ao RAG;
  
## Setup
**Observação**: Testes feitos em uma máquina Linux (Debian based).
**Observação**: Ollama tem suporte limitado par modelos multimodais.

Esse projeto utiliza o python 3.12 para rodar suas dependências:

**Observação**: `py` ou `python` dependendo da sua máquina

`py install 3.12`, ou `winget install Python.Python.3.12` ou direto do site do python

Em seguida, instala utilizando essa versão do Python:

```sh
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Antes, (possível instalar com `spacy` também), seja nescesssário installar o `en_core_web_sm`

`pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl`

Necssário utilizar as bibliotecas em requirements.txt:

`pip install -r requirements.txt`

Além disso, é necessário instalar o Ollama (se Linux):

`curl -fsSL https://ollama.com/install.sh | sh`

Em Windows (PowerShell):

`irm https://ollama.com/install.ps1 | iex`

E ativá-lo em segundo plano:

`ollama serve`

Após isso, é necessário (caso rodado localmente) realizar um *pull* no modelo de preferência. Por testes locais, foi utilizado o modelo qwen2.5 de 0.5 bilhões de parâmetros:

`ollama pull qwen2.5:0.5b`

É fortemente recomendado a utilização de um modelo otimizado para uso de "tools", sendo estes encontrados [aqui](https://ollama.com/search?c=tools).

**Atenção**: Requerimentos de sistema para modelos maiores podem ser custosos. Rode pelo próprio risco.
