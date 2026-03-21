# README

Este arquivo apresenta informações básicas acerca do projeto ["PI-RAG"](https://github.com/igorzeck/PI-RAG/tree/main)

## Organização do projeto

O projeto é organizado da seguinte maneira:

- No diretório **funcoes** encontram-se as funções e o RAG criado para este projeto;
- No diretório **sandbox** encontram-se o playground acessível ao RAG;
  
## Setup
**Observação**: Testes feitos em uma máquina Linux (Debian based).
**Observação**: Ollama tem suporte limitado par modelos multimodais.

Necssário utilizar as bibliotecas em requirements.txt:

`pip install -r requirements.txt`

Além disso, é necessário instalar o Ollama:

`curl -fsSL https://ollama.com/install.sh | sh`

E ativá-lo em segundo plano:

`ollama serve`

Após isso, é necessário (caso rodado localmente) realizar um *pull* no modelo de preferência. Por testes locais, foi utilizado o modelo qwen2.5 de 0.5 bilhões de parâmetros:

`!ollama pull qwen2.5:0.5b`

É fortemente recomendado a utilização de um modelo otimizado para uso de "tools", sendo estes encontrados [aqui](https://ollama.com/search?c=tools).

**Atenção**: Requerimentos de sistema para modelos maiores podem ser custosos. Rode pelo próprio risco.