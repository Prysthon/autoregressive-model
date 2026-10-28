import json
import numpy as np
import pandas as pd


def decisao_prefixo(tabela,matriz,real,quantos_blocos=4,limites=None):
    if limites is None:limites=[32,64,96,127]
    limite=limites[quantos_blocos-1]
    ranking=tabela.copy()
    ranking['nota']=[float(np.mean(abs(matriz[int(i),:limite]-real[:limite]))) for i in ranking['id']]
    ranking=ranking.sort_values(['nota','p','id'])
    melhor=ranking.iloc[0]
    ids=ranking.drop_duplicates('familia').head(3)['id'].astype(int).to_list()
    media=matriz[ids,:limite].mean(axis=0)
    nota_media=float(np.mean(abs(media-real[:limite])))
    if nota_media<float(melhor['nota']):
        return dict(tipo='media_tres_familias',ids=ids,MAE_cv=nota_media)
    return dict(tipo='individual',ids=[int(melhor['id'])],MAE_cv=float(melhor['nota']))


def congelar(tabela,matriz,real,limites=None):
    if limites is None:limites=[32,64,96,127]
    escolha=decisao_prefixo(tabela,matriz,real,4,limites)
    linhas=[];erros=[];erros_incumbente=[]
    for f,(a,b) in enumerate(list(zip(limites[:-1],limites[1:])),start=2):
        e=decisao_prefixo(tabela,matriz,real,f-1,limites)
        pred=matriz[e['ids'],a:b].mean(axis=0)
        mae=float(np.mean(abs(pred-real[a:b])))
        mae_anterior=float(np.mean(abs(matriz[0,a:b]-real[a:b])))
        linhas.append(dict(bloco=f,inicio=1700+120+a,fim=1700+120+b-1,
                           tipo=e['tipo'],ids=json.dumps(e['ids']),MAE=mae,
                           MAE_AR9=mae_anterior,variacao=mae/mae_anterior-1))
        erros.extend(abs(pred-real[a:b]));erros_incumbente.extend(abs(matriz[0,a:b]-real[a:b]))
    resumo=dict(MAE_externo=float(np.mean(erros)),MAE_AR9_externo=float(np.mean(erros_incumbente)))
    # Regra de promoção registrada antes da comparação retrospectiva:
    # pelo menos 2% menos MAE externo agregado e nenhum bloco >20% pior que AR9.
    promover=(resumo['MAE_externo'] <= .98*resumo['MAE_AR9_externo']
              and all(l['variacao']<=.20 for l in linhas))
    resumo['promovido']=bool(promover)
    resumo['regra']='MAE externo ao menos 2% menor; nenhum bloco externo mais de 20% pior que AR9'
    final=escolha if promover else dict(tipo='individual',ids=[0],MAE_cv=float(np.mean(abs(matriz[0]-real))))
    return dict(candidato=escolha,recomendado=final,criterio_promocao=resumo),pd.DataFrame(linhas)


