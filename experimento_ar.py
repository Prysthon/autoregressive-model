
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

PASTA = Path(__file__).parent
RESULTADOS = PASTA / "resultados"

PRIMEIRO_ANO = 1700
MAX_P = 20                     
BLOCOS_VALIDACAO = [(1825, 1864), (1865, 1904), (1905, 1946)]
INICIO_AVALIACAO = 1947        

AZUL, DOURADO, CINZA, VERDE = "#1B3A7A", "#E0A800", "#8C9BBE", "#2E7D8C"


warnings.filterwarnings("ignore", category=FutureWarning)


def indice(ano):
    """Posição do ano no vetor de dados (1700 -> 0, 1701 -> 1, ...)."""
    return ano - PRIMEIRO_ANO


# ----- 1. Dados -----

def carregar_dados():
    dados = pd.read_csv(PASTA / "dados" / "sunspots.csv")
    anos = dados["YEAR"].astype(int).to_numpy()
    y = dados["SUNACTIVITY"].to_numpy(dtype=float)
    # Integridade: 309 anos seguidos, de 1700 a 2008, sem valores faltando.
    assert list(anos) == list(range(1700, 2009)), "Os anos precisam ir de 1700 a 2008, sem buracos."
    assert not np.isnan(y).any(), "Há valores faltando na série."
    return y


# ----- 2. O modelo AR(p) -----
#    previsão = c + phi_1 * y[t-1] + phi_2 * y[t-2] + ... + phi_p * y[t-p]

def montar_tabela(y, p):
    """Monta as entradas e os alvos do treino.

    Cada linha de X é [1, y[t-1], ..., y[t-p]] e o alvo é y[t].
    O 1 serve para o intercepto c. Todos os modelos começam no mesmo
    alvo (t = 20) para que as 20 ordens sejam comparadas de forma justa.
    """
    X, alvos = [], []
    for t in range(MAX_P, len(y)):
        X.append([1.0] + [y[t - k] for k in range(1, p + 1)])
        alvos.append(y[t])
    return np.array(X), np.array(alvos)


def treinar(y, p):
    """Mínimos quadrados: os pesos que minimizam a soma dos erros ao quadrado."""
    X, alvos = montar_tabela(y, p)
    pesos = np.linalg.lstsq(X, alvos, rcond=None)[0]
    return pesos                    # [c, phi_1, ..., phi_p]


def prever(pesos, historico):
    """Prevê o próximo valor usando só o passado."""
    p = len(pesos) - 1
    ultimos = [historico[-k] for k in range(1, p + 1)]   # y[t-1], y[t-2], ...
    return pesos[0] + np.dot(pesos[1:], ultimos)


def prever_varios_anos(pesos, historico, anos):
    """Previsão recursiva: cada previsão vira entrada da próxima."""
    memoria = list(historico)
    previsoes = []
    for _ in range(anos):
        proxima = prever(pesos, memoria)
        previsoes.append(proxima)
        memoria.append(proxima)
    return previsoes


# ----- 3. Métricas -----

def mae(real, previsto):
    return np.mean(np.abs(np.array(real) - np.array(previsto)))


def rmse(real, previsto):
    return np.sqrt(np.mean((np.array(real) - np.array(previsto)) ** 2))


def vies(real, previsto):
    return np.mean(np.array(previsto) - np.array(real))


# ----- 4. Escolha de p com validação temporal (só usa anos até 1946) -----

def escolher_p(y):
    mae_por_p = {}
    for p in range(1, MAX_P + 1):
        erros = []
        for inicio, fim in BLOCOS_VALIDACAO:
            pesos = treinar(y[:indice(inicio)], p)        # treina só com os anos antes do bloco
            for ano in range(inicio, fim + 1):
                t = indice(ano)
                erros.append(abs(y[t] - prever(pesos, y[:t])))
        mae_por_p[p] = np.mean(erros)
    melhor_p = min(mae_por_p, key=mae_por_p.get)
    return melhor_p, mae_por_p


# ----- 5. Avaliação de 1947 a 2008, um ano à frente -----

def avaliar(y, pesos):
    """Para cada ano: prever com o passado e só depois olhar o valor real."""
    linhas = []
    for ano in range(INICIO_AVALIACAO, 2009):
        t = indice(ano)
        linhas.append({"ano": ano,
                       "real": y[t],
                       "AR": prever(pesos, y[:t]),
                       "persistencia": y[t - 1]})    # referência: repetir o ano anterior
    return pd.DataFrame(linhas)


def erro_por_horizonte(y, pesos):
    """Erro ao prever h anos à frente, nas mesmas 53 origens (1946 a 1998)."""
    linhas = []
    for h in [1, 5, 10]:
        reais, ar, persistencia = [], [], []
        for origem in range(1946, 1999):
            t = indice(origem)
            ar.append(prever_varios_anos(pesos, y[:t + 1], h)[-1])
            reais.append(y[t + h])
            persistencia.append(y[t])
        linhas.append({"horizonte": h, "MAE_AR": mae(reais, ar),
                       "MAE_persistencia": mae(reais, persistencia)})
    return pd.DataFrame(linhas)


def diagnosticar(y, pesos):
    p = len(pesos) - 1
    treino = y[:indice(INICIO_AVALIACAO)]
    X, alvos = montar_tabela(treino, p)
    residuos = alvos - X @ pesos
    raizes = np.roots(np.r_[-pesos[1:][::-1], 1.0])
    return {
        "ADF (p-valor)": adfuller(treino, regression="c", autolag="AIC")[1],
        "Menor módulo das raízes": np.min(np.abs(raizes)),
        "Ljung-Box (p-valor)": acorr_ljungbox(residuos, lags=[20], model_df=p).lb_pvalue.iloc[0],
    }


# ----- 6. Demonstração: reconstruir uma previsão feita no passado -----

def demonstracao(y, pesos, origem, horizonte):
    t0 = indice(origem) + 1                         # primeiro ano previsto
    anos = list(range(origem + 1, origem + horizonte + 1))
    tabela = pd.DataFrame({
        "ano": anos,
        "real": y[t0:t0 + horizonte],
        "recursivo": prever_varios_anos(pesos, y[:t0], horizonte),          # não recebe reais novos
        "um_passo": [prever(pesos, y[:t0 + k]) for k in range(horizonte)],  # recebe os reais
    })
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(anos, tabela.real, "o-", color=AZUL, label="Real")
    ax.plot(anos, tabela.recursivo, "o--", color=DOURADO, label="Recursivo, sem novos reais")
    ax.plot(anos, tabela.um_passo, "o-", color=VERDE, label="Um passo, com novos reais")
    ax.set(xlabel="Ano", ylabel="Índice de atividade solar", title=f"Previsão feita em {origem}")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTADOS / f"demo_{origem}.png", dpi=180)
    tabela.to_csv(RESULTADOS / f"demo_{origem}.csv", index=False)
    return tabela


# ----- Gráficos do experimento -----

def grafico_escolha_p(mae_por_p, melhor_p):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(list(mae_por_p), list(mae_por_p.values()), "o-", color=AZUL)
    ax.plot(melhor_p, mae_por_p[melhor_p], "o", markersize=12, color=DOURADO)
    ax.set(xlabel="Ordem p", ylabel="MAE na validação", xticks=range(1, MAX_P + 1),
           title="Escolha de p")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "selecao.png", dpi=180)


def grafico_previsoes(previsoes, p):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(previsoes.ano, previsoes.real, color=AZUL, label="Real")
    ax.plot(previsoes.ano, previsoes.AR, color=DOURADO, label=f"AR({p})")
    ax.plot(previsoes.ano, previsoes.persistencia, color=CINZA, label="Persistência")
    ax.set(xlabel="Ano previsto", ylabel="Índice de atividade solar",
           title="Previsões de um ano à frente, 1947 a 2008")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(RESULTADOS / "previsoes.png", dpi=180)


# ----- Programa principal -----

def titulo(texto):
    print(f"\n=== {texto} ===")


def main():
    parser = argparse.ArgumentParser(description="Previsão anual da atividade solar com AR(p).")
    parser.add_argument("--origem", type=int, default=1954, help="último ano conhecido (1946 a 2007)")
    parser.add_argument("--horizonte", type=int, default=10, help="quantos anos prever")
    parser.add_argument("--so-demo", action="store_true", help="mostra só a demonstração")
    args = parser.parse_args()
    if not 1946 <= args.origem <= 2007 or args.horizonte < 1 or args.origem + args.horizonte > 2008:
        parser.error("use origem entre 1946 e 2007, com origem + horizonte até 2008.")

    RESULTADOS.mkdir(exist_ok=True)
    y = carregar_dados()


    p, mae_por_p = escolher_p(y)
    pesos = treinar(y[:indice(INICIO_AVALIACAO)], p)

    if not args.so_demo:
        titulo("1. Dados")
        print(f"{len(y)} anos, de 1700 a 2008. Treino até 1946, avaliação de 1947 a 2008.")

        titulo("2. Escolha de p (menor MAE na validação)")
        selecao = pd.Series(mae_por_p, name="MAE_validacao").rename_axis("p").sort_values()
        print(selecao.head(5).round(3).to_string())
        print(f"Escolhido: AR({p})")
        selecao.to_csv(RESULTADOS / "selecao.csv")
        grafico_escolha_p(mae_por_p, p)

        titulo("3. Pesos aprendidos (treino de 1700 a 1946)")
        nomes = ["c"] + [f"phi_{k}" for k in range(1, p + 1)]
        print(pd.Series(pesos, index=nomes).round(4).to_string())

        titulo("4. Avaliação de um ano à frente, 1947 a 2008")
        previsoes = avaliar(y, pesos)
        metricas = pd.DataFrame({
            "MAE": [mae(previsoes.real, previsoes[m]) for m in ["persistencia", "AR"]],
            "RMSE": [rmse(previsoes.real, previsoes[m]) for m in ["persistencia", "AR"]],
            "Viés": [vies(previsoes.real, previsoes[m]) for m in ["persistencia", "AR"]],
        }, index=["Persistência", f"AR({p})"])
        print(metricas.round(3).to_string())
        ganho = 100 * (1 - metricas.MAE.iloc[1] / metricas.MAE.iloc[0])
        print(f"O AR({p}) teve {ganho:.1f}% menos MAE que a persistência.")
        previsoes.to_csv(RESULTADOS / "previsoes.csv", index=False)
        metricas.to_csv(RESULTADOS / "metricas.csv")
        grafico_previsoes(previsoes, p)

        titulo("5. Maior erro")
        previsoes["erro_absoluto"] = (previsoes.real - previsoes.AR).abs()
        print(previsoes[previsoes.ano.between(1955, 1958)].round(2).to_string(index=False))

        titulo("6. Erro por horizonte (53 origens, 1946 a 1998)")
        horizontes = erro_por_horizonte(y, pesos)
        print(horizontes.round(3).to_string(index=False))
        horizontes.to_csv(RESULTADOS / "horizontes.csv", index=False)

        titulo("7. Diagnóstico")
        for nome, valor in diagnosticar(y, pesos).items():
            print(f"{nome}: {valor:.4f}")

    titulo(f"Demonstração: previsão feita em {args.origem}, {args.horizonte} anos à frente")
    print(demonstracao(y, pesos, args.origem, args.horizonte).round(2).to_string(index=False))

    print("\nArquivos salvos em resultados/. Feche as janelas dos gráficos para encerrar.")
    plt.show()


if __name__ == "__main__":
    main()