
# Otimização de Layout de Galpão com Algoritmo Genético

Projeto desenvolvido na disciplina de **Inteligência Artificial**, aplicando Algoritmos Genéticos (IA Clássica não-conexionista) para otimizar a disposição de produtos em um galpão de armazenamento.

## 📌 Problema

Dado um galpão com várias posições de armazenamento e produtos com diferentes frequências de saída, encontrar a melhor disposição dos produtos para **minimizar o deslocamento dos funcionários** durante a separação de pedidos, respeitando restrições de **peso** e **dimensões** de cada posição.

Ver descrição completa em [PROBLEMA.md](PROBLEMA.md).

## 🧬 Por que Algoritmo Genético?

Para 20 posições, existem `20!` (≈ 2,4 quintilhões) de arranjos possíveis. Testar todas as combinações por força bruta é inviável — o AG explora o espaço de busca de forma inteligente, evoluindo soluções através de seleção, cruzamento e mutação.

## 📂 Estrutura

```
.
├── data/
│   ├── dim_produtos.csv     # cadastro de produtos (peso, dimensões)
│   ├── dim_posicoes.csv     # cadastro de posições (capacidade, distância)
│   └── fato_saidas.csv      # registros de saída (90 dias simulados)
├── ag_galpao.py             # implementação completa do AG
├── PROBLEMA.md              # descrição do problema
└── README.md
```

## 🗃️ Modelagem dos dados

Modelo dimensional inspirado em star schema:

- **`dim_produtos`** — atributos físicos dos produtos
- **`dim_posicoes`** — atributos das posições do galpão
- **`fato_saidas`** — eventos de saída (a frequência é calculada a partir dessa tabela)

## ⚙️ Componentes do AG

| Componente | Implementação |
|------------|---------------|
| Representação | Permutação de produtos por posição |
| Fitness | `Σ (frequência × distância) + penalidades` |
| Seleção | Torneio (k=3) |
| Cruzamento | Order Crossover (OX) |
| Mutação | Swap (troca de duas posições) |
| Elitismo | 2 melhores passam direto pra próxima geração |

## 🚀 Como rodar

```bash
# instalar dependências
pip install pandas numpy matplotlib

# rodar o AG
python ag_galpao.py
```

Saídas geradas:
- `evolucao_ag.png` — gráfico da evolução do fitness
- `layout_otimizado.csv` — layout final encontrado

## 📊 Parâmetros

Os parâmetros do AG podem ser ajustados no topo de `ag_galpao.py`:

```python
TAMANHO_POPULACAO = 100
NUM_GERACOES = 500
TAXA_CRUZAMENTO = 0.85
TAXA_MUTACAO = 0.10
K_TORNEIO = 3
TAMANHO_ELITE = 2
PENALIDADE = 100000
```

## 🛠️ Stack

- Python 3
- pandas, numpy
- matplotlib

## 📚 Disciplina

Inteligência Artificial — IA Clássica (não conexionista): Buscas, IA Simbólica, Satisfação de Restrições, Algoritmos Genéticos, Redes Bayesianas.
