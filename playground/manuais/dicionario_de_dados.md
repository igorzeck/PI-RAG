# Dicionário de Dados - Base de Churn

Este documento explica as colunas que podem ser encontradas no dataset `base_clientes_churn.csv` localizado na pasta de datasets do data warehouse (ou no diretório de playground). 

## Tabela: Clientes

| Nome da Coluna | Tipo de Dado | Descrição |
| --- | --- | --- |
| `ID_Cliente` | Texto | Identificador único e anonimizado do cliente na plataforma. (Ex: CLI-1002). |
| `Plano` | Categórico | Indica a camada da assinatura ativa do usuário no momento da exportação (Ex: Básico, Plus, Premium). |
| `Meses_Ativo` | Numérico | Tempo de vida (tenure) do cliente medido em número de faturas mensais pagas desde a criação da conta. |
| `Chamados_Suporte` | Numérico | Contagem total agregada de ligações, tickets e chamados de suporte técnico abertos pelo cliente em toda a vida da assinatura. |
| `Mensalidade` | Numérico (Float) | O valor monetário (em BRL/R$) cobrado mensalmente. Pode refletir descontos aplicados. |
| `Status` | Categórico | Variável alvo do Churn: `Ativo` significa que o serviço está funcionando. `Cancelado` significa que o cliente rescindiu o contrato. |
| `Motivo_Cancelamento` | Texto | Caso `Status` seja `Cancelado`, esta coluna trará a razão mapeada pela equipe de retenção (Ex. Preço, Concorrente). Caso `Status` seja `Ativo`, a coluna será vázia. |

## Observações para os Analistas de Dados:
- Ao calcular a métrica estatística de probabilidade de cancelamento (Churn Rate), certifique-se de segmentar por `Plano`, pois as taxas comportamentais variam severamente.
- Clientes com zero `Meses_Ativo` representam early-churn (desistiram antes do fechamento da primeira fatura, usualmente no prazo de 7 dias de arrependimento de compra).
