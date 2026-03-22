from funcoes.rag import agente
resp = agente.invoke({"messages": [{"role": "user", "content": "Liste os arquivos existentes no diretório."}]}, {"configurable": {"thread_id": "99", "recursion_limit": 5}})
for msg in resp['messages']:
    if hasattr(msg, 'tool_calls'):
        print('Tool calls:', msg.tool_calls)
    print(msg.content)
