"""Refaz a investigação temporal: uv run --locked python prever_manchas.py."""
from pathlib import Path
from investigacao_manchas import investigar, exportar, graficos, relatorio


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--avancado',action='store_true',help='Executar investigação não linear adicional')
    parser.add_argument('--refazer-busca',action='store_true',help='Ignorar cache da investigação avançada')
    args=parser.parse_args()
    pasta=Path(__file__).resolve().parent
    if args.avancado:
        from manchas_avaliacao_avancada import executar as avancado, exportar as excel, relatorio as texto, graficos as figuras
        e=avancado(pasta,refazer=args.refazer_busca,progresso=True)
        saida=excel(e,pasta);rel=texto(e,pasta);figuras(e,pasta)
        m=e['comparacao'].copy();m[['MAPE','WAPE']]*=100
        print(m.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
        print('AR9 permanece como padrão. Experimental melhora MAPE, mas piora MAE e RMSE.')
        print('Excel:',saida);print('Relatório:',rel)
        return
    e=investigar(pasta,progresso=True)
    saida=exportar(e,pasta)
    graficos(e,pasta)
    texto=relatorio(e,pasta)
    print('\nConfiguração escolhida só no desenvolvimento:',e['escolhido'])
    m=e['comparacao'].copy()
    m[['MAPE','WAPE']]*=100
    print('\nComparação retrospectiva de 1947–2008 (já observado):')
    print(m.to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('\nValidação aninhada:')
    print(e['blocos_aninhados'].to_string(index=False))
    print('\nExcel:',saida)
    print('Relatório:',texto)


if __name__=='__main__':
    main()
