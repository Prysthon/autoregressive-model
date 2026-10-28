"""Seleção usa exclusivamente os anos anteriores a 1947."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline



def transformar(y, nome):
    if nome == 'raiz': return np.sqrt(y)
    if nome == 'log1p': return np.log1p(y)
    return y


def ajustar_prever(historico, c, modelo=None, hold_back=30):
    z = transformar(np.asarray(historico), c['transformacao'])
    p = c['p']
    if modelo is None:
        indices = np.arange(hold_back, len(z))
        x = z[indices[:, None] - np.arange(1, p+1)]
        reg = LinearRegression() if c['alpha'] == 0 else Ridge(alpha=c['alpha'])
        modelo = make_pipeline(StandardScaler(), reg).fit(x, z[indices])
    pred = float(modelo.predict(z[-np.arange(1, p+1)][None, :])[0])
    if c['transformacao'] == 'raiz': pred = max(0, pred) ** 2
    elif c['transformacao'] == 'log1p': pred = np.expm1(pred)
    return max(0, pred), modelo


def prever_bloco(y, inicio, fim, c):
    modelo = None
    previsoes = []
    for i in range(inicio, fim):
        reutilizar = modelo if c['atualizacao'] == 'fixo' else None
        pred, modelo = ajustar_prever(y[:i], c, reutilizar)
        previsoes.append(pred)
    return np.array(previsoes)


def selecionar(y, progresso=False):
    if len(y) != 247:
        raise ValueError('Seleção espera apenas os 247 valores anteriores a 1947.')
    folds = [(120,152),(152,184),(184,216),(216,247)]
    candidatos = [dict(transformacao=t,p=p,alpha=a,atualizacao=u)
                  for t in ['original','raiz','log1p']
                  for p in [2,3,5,9,11,15,19]
                  for a in [0,1,10]
                  for u in ['fixo','expansivo']]
    linhas=[]; preds={}
    for k,c in enumerate(candidatos):
        erros=[]; maes=[]; quadro=[]
        for f,(inicio,fim) in enumerate(folds):
            pred=prever_bloco(y,inicio,fim,c)
            real=y[inicio:fim]
            erros.extend(pred-real)
            maes.append(np.mean(np.abs(pred-real)))
            quadro.extend([{'indice':i,'fold':f+1,'real':float(v),'predito':float(p)}
                           for i,v,p in zip(range(inicio,fim),real,pred)])
        linhas.append({'candidato':k,**c,'MAE_cv':np.mean(np.abs(erros)),
                       'RMSE_cv':np.sqrt(np.mean(np.square(erros))),
                       'MAPE_cv':np.mean(np.abs(np.asarray(erros))/y[120:247]),
                       'MAE_folds_std':np.std(maes,ddof=1),
                       **{f'MAE_fold_{j+1}':v for j,v in enumerate(maes)}})
        preds[k]=quadro
        if progresso and k%30==0: print('Candidatos avaliados:',k+1,flush=True)
    tabela=pd.DataFrame(linhas).sort_values(['MAE_cv','p','alpha']).reset_index(drop=True)
    # Critério definido antes da comparação retrospectiva: menor MAE temporal médio.
    melhor=tabela.iloc[0].to_dict()
    c={key:melhor[key] for key in ['transformacao','p','alpha','atualizacao']}
    c['p']=int(c['p']);c['alpha']=float(c['alpha'])
    return c,tabela,pd.DataFrame(preds[int(melhor['candidato'])]),preds


