"""Executar com: uv run --locked python prever_manchas.py"""
from pathlib import Path
from manchas_solares import executar, exportar, gerar_graficos


def main():
    pasta = Path(__file__).resolve().parent
    e = executar(pasta)
    saida = exportar(e, pasta)
    gerar_graficos(e, pasta)
    print(e['configuracao'].iloc[[0, 4, 5, 6, 7]].to_string(index=False))
    m = e['metricas'].copy()
    m['MAPE (%)'] *= 100
    print('\nMétricas de previsões de um ano à frente no teste:')
    print(m.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nExcel:', saida)
    print('Gráficos:', saida.parent)


if __name__ == '__main__':
    main()
