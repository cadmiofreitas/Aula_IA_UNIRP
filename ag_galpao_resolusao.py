"""
================================================================================
ALGORITMO GENETICO - OTIMIZACAO DO LAYOUT DE ESTOQUE DE UM GALPAO
================================================================================

Objetivo: encontrar a melhor disposicao de produtos nas posicoes do galpao
de modo a minimizar o deslocamento dos funcionarios ate a area de expedicao.

Restricoes consideradas:
    - Peso maximo suportado por cada posicao
    - Dimensoes maximas (altura, largura, profundidade) de cada posicao

Tecnica: Algoritmo Genetico classico com:
    - Representacao: permutacao de produtos por posicao
    - Selecao: torneio (k=3)
    - Cruzamento: Order Crossover (OX)
    - Mutacao: swap (troca de duas posicoes)
    - Elitismo: 2 melhores passam direto pra proxima geracao

Autor: Cadmio
Disciplina: Inteligencia Artificial
================================================================================
"""

import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURACOES
# ==============================================================================

DATA_DIR = "data"  # diretorio onde os CSVs estao
SEED = 42

# Parametros do AG
TAMANHO_POPULACAO = 100
NUM_GERACOES = 500
TAXA_CRUZAMENTO = 0.85
TAXA_MUTACAO = 0.10
K_TORNEIO = 3
TAMANHO_ELITE = 2
PENALIDADE = 100000  # custo adicionado por cada violacao de restricao


# ==============================================================================
# 1. CARREGAR DADOS
# ==============================================================================

def carregar_dados(data_dir=DATA_DIR):
    """Carrega os 3 CSVs e retorna os DataFrames."""
    produtos = pd.read_csv(os.path.join(data_dir, "dim_produtos.csv"))
    posicoes = pd.read_csv(os.path.join(data_dir, "dim_posicoes.csv"))
    fato_saidas = pd.read_csv(os.path.join(data_dir, "fato_saidas.csv"))

    # Calcular frequencia de saida por produto a partir do fato
    freq_saida = fato_saidas.groupby("id_produto")["quantidade"].sum().to_dict()
    produtos["freq_saida"] = produtos["id_produto"].map(freq_saida)

    return produtos, posicoes, fato_saidas


def montar_dicionarios(produtos, posicoes):
    """
    Monta dicionarios pra acesso rapido durante o calculo do fitness.
    Acessar dict e MUITO mais rapido que df.loc[] dentro de loops.
    """
    dicts = {
        "freq": dict(zip(produtos["id_produto"], produtos["freq_saida"])),
        "peso": dict(zip(produtos["id_produto"], produtos["peso_kg"])),
        "altura": dict(zip(produtos["id_produto"], produtos["altura_cm"])),
        "largura": dict(zip(produtos["id_produto"], produtos["largura_cm"])),
        "comp": dict(zip(produtos["id_produto"], produtos["comprimento_cm"])),
        "dist": dict(zip(posicoes["id_posicao"], posicoes["dist_porta"])),
        "peso_max": dict(zip(posicoes["id_posicao"], posicoes["peso_max_kg"])),
        "altura_max": dict(zip(posicoes["id_posicao"], posicoes["altura_max_cm"])),
        "largura_max": dict(zip(posicoes["id_posicao"], posicoes["largura_max_cm"])),
        "prof": dict(zip(posicoes["id_posicao"], posicoes["profundidade_cm"])),
    }
    return dicts


# ==============================================================================
# 2. CROMOSSOMO (REPRESENTACAO)
# ==============================================================================
# Um cromossomo e uma lista de produtos onde o INDICE da lista corresponde
# a posicao em ids_posicoes.
# Exemplo: cromossomo[0] = produto que esta na posicao ids_posicoes[0]

def gerar_cromossomo_aleatorio(ids_produtos):
    """Gera um cromossomo embaralhando os produtos."""
    cromossomo = ids_produtos.copy()
    random.shuffle(cromossomo)
    return cromossomo


def gerar_populacao_inicial(tamanho, ids_produtos):
    """Cria uma populacao inicial de cromossomos aleatorios."""
    return [gerar_cromossomo_aleatorio(ids_produtos) for _ in range(tamanho)]


# ==============================================================================
# 3. FUNCAO DE FITNESS
# ==============================================================================
# Quanto MENOR o fitness, MELHOR o cromossomo.
# fitness = deslocamento_total + penalidades_por_violacao_de_restricao

def calcular_fitness(cromossomo, ids_posicoes, dicts):
    """
    Calcula o custo total de um layout.
    Soma:
        - Deslocamento: freq_saida x dist_porta pra cada par produto-posicao
        - Penalidades: por cada violacao de restricao fisica
    """
    deslocamento = 0
    penalidades = 0

    for i, id_prod in enumerate(cromossomo):
        id_pos = ids_posicoes[i]

        # Deslocamento (objetivo principal)
        deslocamento += dicts["freq"][id_prod] * dicts["dist"][id_pos]

        # Restricao de peso
        if dicts["peso"][id_prod] > dicts["peso_max"][id_pos]:
            penalidades += PENALIDADE

        # Restricoes dimensionais
        if dicts["altura"][id_prod] > dicts["altura_max"][id_pos]:
            penalidades += PENALIDADE
        if dicts["largura"][id_prod] > dicts["largura_max"][id_pos]:
            penalidades += PENALIDADE
        if dicts["comp"][id_prod] > dicts["prof"][id_pos]:
            penalidades += PENALIDADE

    return deslocamento + penalidades


# ==============================================================================
# 4. SELECAO POR TORNEIO
# ==============================================================================

def selecao_torneio(populacao, fitness_lista, k=K_TORNEIO):
    """
    Sorteia k cromossomos aleatorios e retorna o melhor (menor fitness).
    """
    indices = random.sample(range(len(populacao)), k)
    melhor_idx = min(indices, key=lambda idx: fitness_lista[idx])
    return populacao[melhor_idx]


# ==============================================================================
# 5. CRUZAMENTO (ORDER CROSSOVER - OX)
# ==============================================================================
# OX preserva a propriedade de permutacao (cada produto aparece exatamente 1 vez).

def cruzamento_ox(pai1, pai2):
    """
    Order Crossover (OX):
    1. Sorteia 2 pontos de corte
    2. Copia o trecho entre os cortes do Pai 1 pro Filho
    3. Preenche o resto com elementos do Pai 2, na ordem, pulando duplicatas
    """
    tamanho = len(pai1)

    ponto1 = random.randint(0, tamanho - 2)
    ponto2 = random.randint(ponto1 + 1, tamanho - 1)

    filho = [None] * tamanho
    filho[ponto1:ponto2 + 1] = pai1[ponto1:ponto2 + 1]
    ja_no_filho = set(filho[ponto1:ponto2 + 1])

    # Percorrer Pai 2 a partir do corte final, dando a volta
    pai2_ordenado = pai2[ponto2 + 1:] + pai2[:ponto2 + 1]
    pai2_filtrado = [x for x in pai2_ordenado if x not in ja_no_filho]

    idx_filho = (ponto2 + 1) % tamanho
    for valor in pai2_filtrado:
        while filho[idx_filho] is not None:
            idx_filho = (idx_filho + 1) % tamanho
        filho[idx_filho] = valor

    return filho


# ==============================================================================
# 6. MUTACAO (SWAP)
# ==============================================================================
# Sorteia 2 posicoes do cromossomo e troca os produtos entre elas.
# Preserva a permutacao naturalmente.

def mutacao_swap(cromossomo, taxa=TAXA_MUTACAO):
    """Com probabilidade `taxa`, troca dois elementos aleatorios do cromossomo."""
    crom = cromossomo.copy()
    if random.random() < taxa:
        i, j = random.sample(range(len(crom)), 2)
        crom[i], crom[j] = crom[j], crom[i]
    return crom


# ==============================================================================
# 7. LOOP PRINCIPAL DO AG
# ==============================================================================

def algoritmo_genetico(ids_produtos, ids_posicoes, dicts,
                       tamanho_pop=TAMANHO_POPULACAO,
                       num_geracoes=NUM_GERACOES,
                       taxa_cruzamento=TAXA_CRUZAMENTO,
                       taxa_mutacao=TAXA_MUTACAO,
                       k_torneio=K_TORNEIO,
                       tamanho_elite=TAMANHO_ELITE,
                       verbose=True):
    """
    Executa o AG e retorna o melhor cromossomo encontrado + historico de fitness.
    """
    # Populacao inicial
    populacao = gerar_populacao_inicial(tamanho_pop, ids_produtos)
    fitness_lista = [calcular_fitness(c, ids_posicoes, dicts) for c in populacao]

    historico_melhor = []
    historico_media = []

    for geracao in range(num_geracoes):
        # Ordenar populacao pelo fitness
        ordenados = sorted(range(len(populacao)), key=lambda idx: fitness_lista[idx])

        # ------------------------------------------------------------
        # ELITISMO: melhores cromossomos passam direto
        # ------------------------------------------------------------
        nova_populacao = [populacao[idx] for idx in ordenados[:tamanho_elite]]

        # ------------------------------------------------------------
        # Gerar resto da populacao por selecao + cruzamento + mutacao
        # ------------------------------------------------------------
        while len(nova_populacao) < tamanho_pop:
            pai1 = selecao_torneio(populacao, fitness_lista, k_torneio)
            pai2 = selecao_torneio(populacao, fitness_lista, k_torneio)

            # Cruzamento (com probabilidade taxa_cruzamento)
            if random.random() < taxa_cruzamento:
                filho = cruzamento_ox(pai1, pai2)
            else:
                filho = pai1.copy()

            # Mutacao
            filho = mutacao_swap(filho, taxa_mutacao)

            nova_populacao.append(filho)

        # Atualizar populacao e fitness
        populacao = nova_populacao
        fitness_lista = [calcular_fitness(c, ids_posicoes, dicts) for c in populacao]

        # Estatisticas
        melhor_fit = min(fitness_lista)
        media_fit = np.mean(fitness_lista)
        historico_melhor.append(melhor_fit)
        historico_media.append(media_fit)

        if verbose and (geracao % 50 == 0 or geracao == num_geracoes - 1):
            print(f"Geracao {geracao:>4} | Melhor: {melhor_fit:>10,.0f} | Media: {media_fit:>10,.0f}")

    # Retornar o melhor cromossomo encontrado
    melhor_idx = fitness_lista.index(min(fitness_lista))
    melhor_cromossomo = populacao[melhor_idx]
    melhor_fitness = fitness_lista[melhor_idx]

    return melhor_cromossomo, melhor_fitness, historico_melhor, historico_media


# ==============================================================================
# 8. UTILIDADES (VISUALIZACAO E COMPARACAO)
# ==============================================================================

def montar_layout(cromossomo, ids_posicoes, produtos):
    """Monta um DataFrame mostrando o layout final."""
    layout = pd.DataFrame({
        "id_posicao": ids_posicoes,
        "id_produto": cromossomo,
    })
    layout = layout.merge(
        produtos[["id_produto", "nome", "peso_kg", "freq_saida"]],
        on="id_produto"
    )
    return layout


def calcular_solucao_ingenua(produtos, posicoes, ids_posicoes, dicts):
    """
    Solucao baseline: ordena produtos por freq decrescente
    e posicoes por dist crescente (sem considerar restricoes).
    Serve de referencia pra comparar com o AG.
    """
    prods_ord = produtos.sort_values("freq_saida", ascending=False)["id_produto"].tolist()
    poss_ord = posicoes.sort_values("dist_porta", ascending=True)["id_posicao"].tolist()

    crom = [None] * len(ids_posicoes)
    for prod, pos in zip(prods_ord, poss_ord):
        idx = ids_posicoes.index(pos)
        crom[idx] = prod

    fit = calcular_fitness(crom, ids_posicoes, dicts)
    return crom, fit


def plotar_evolucao(historico_melhor, historico_media, fit_ingenuo=None,
                    salvar_em="evolucao_ag.png"):
    """Plota a evolucao do fitness ao longo das geracoes."""
    plt.figure(figsize=(10, 5))
    plt.plot(historico_melhor, label="Melhor fitness", linewidth=2)
    plt.plot(historico_media, label="Fitness medio", alpha=0.7)
    if fit_ingenuo is not None:
        plt.axhline(y=fit_ingenuo, color="red", linestyle="--",
                    label=f"Solucao ingenua ({fit_ingenuo:,.0f})")
    plt.xlabel("Geracao")
    plt.ylabel("Fitness (custo) - menor e melhor")
    plt.title("Evolucao do AG - Otimizacao do Layout do Galpao")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(salvar_em, dpi=100)
    plt.close()
    print(f"\nGrafico salvo em: {salvar_em}")


# ==============================================================================
# 9. EXECUCAO PRINCIPAL
# ==============================================================================

def main():
    # Reprodutibilidade
    np.random.seed(SEED)
    random.seed(SEED)

    # 1. Carregar dados
    print("=" * 70)
    print("CARREGANDO DADOS...")
    print("=" * 70)
    produtos, posicoes, fato_saidas = carregar_dados()
    dicts = montar_dicionarios(produtos, posicoes)

    ids_produtos = produtos["id_produto"].tolist()
    ids_posicoes = posicoes["id_posicao"].tolist()

    print(f"  {len(produtos)} produtos | {len(posicoes)} posicoes | "
          f"{len(fato_saidas)} registros de saida")

    # 2. Calcular solucao ingenua (baseline)
    print("\n" + "=" * 70)
    print("BASELINE - Solucao Ingenua (ordenar por freq vs dist)")
    print("=" * 70)
    crom_ing, fit_ing = calcular_solucao_ingenua(
        produtos, posicoes, ids_posicoes, dicts
    )
    print(f"  Fitness ingenuo: {fit_ing:,}")

    # 3. Rodar o AG
    print("\n" + "=" * 70)
    print("RODANDO ALGORITMO GENETICO")
    print("=" * 70)
    print(f"  Populacao: {TAMANHO_POPULACAO} | Geracoes: {NUM_GERACOES}")
    print(f"  Taxa cruzamento: {TAXA_CRUZAMENTO} | Taxa mutacao: {TAXA_MUTACAO}")
    print(f"  Torneio K={K_TORNEIO} | Elite: {TAMANHO_ELITE}\n")

    melhor_crom, melhor_fit, hist_melhor, hist_media = algoritmo_genetico(
        ids_produtos, ids_posicoes, dicts
    )

    # 4. Resultados
    print("\n" + "=" * 70)
    print("RESULTADOS FINAIS")
    print("=" * 70)
    print(f"  Solucao ingenua:  {fit_ing:>10,}")
    print(f"  Solucao do AG:    {melhor_fit:>10,}")
    melhoria = (1 - melhor_fit / fit_ing) * 100
    print(f"  Melhoria do AG vs ingenua: {melhoria:.2f}%")

    # 5. Layout final
    print("\n" + "=" * 70)
    print("LAYOUT OTIMIZADO PELO AG")
    print("=" * 70)
    layout = montar_layout(melhor_crom, ids_posicoes, produtos)
    layout = layout.merge(posicoes[["id_posicao", "dist_porta", "peso_max_kg"]],
                          on="id_posicao")
    layout = layout.sort_values("dist_porta")
    print(layout.to_string(index=False))

    # 6. Salvar grafico
    plotar_evolucao(hist_melhor, hist_media, fit_ingenuo=fit_ing)

    # 7. Salvar layout final em CSV
    layout.to_csv("layout_otimizado.csv", index=False)
    print("\nLayout otimizado salvo em: layout_otimizado.csv")


if __name__ == "__main__":
    main()
