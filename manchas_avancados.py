"""Candidatos causais. A seleção recebe exclusivamente 1700–1946."""
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

FOLDS=[(120,152),(152,184),(184,216),(216,247)]


class Regimes(BaseEstimator,TransformerMixin):
    def __init__(self,modo='fase'):
        self.modo=modo
    def fit(self,x,y=None):
        self.limiar_=float(np.median(x[:,0]))
        return self
    def transform(self,x):
        gate=(x[:,0]>x[:,1]) if self.modo=='fase' else (x[:,0]>self.limiar_)
        g=gate.astype(float)[:,None]
        return np.hstack([x,x*g,g])


def candidatos():
    lista=[]
    def add(familia='ridge',p=9,t='raiz',**kw):
        c=dict(familia=familia,lags=list(range(1,p+1)),transformacao=t,
               alpha=1.0,janela=0,meia_vida=0,alvo='nivel')
        c.update(kw);lista.append(c)
    # Primeiro candidato reproduz o vencedor da investigação anterior.
    add()
    for janela in [60,100,150]: add(janela=janela)
    for meia_vida in [25,50,100]:add(meia_vida=meia_vida)
    for janela in [100,150]:add(janela=janela,meia_vida=50)
    for t in ['original','raiz']:
        for p in [3,5,9]:
            for alpha in [1,10,100]:
                add('quadratico',p,t,alpha=alpha)
                add('spline',p,t,alpha=alpha)
            for alpha in [.1,1,10]:
                for gamma in [.1,.5]:add('kernel_rbf',p,t,alpha=alpha,gamma=gamma)
            for custo in [1,10]:
                for gamma in [.1,.5]:add('svr',p,t,C=custo,gamma=gamma)
            for alpha in [1,10]:
                add('regime_fase',p,t,alpha=alpha)
                add('regime_nivel',p,t,alpha=alpha)
                add('ridge',p,t,alpha=alpha,alvo='diferenca')
            for folhas in [3,7]:add('boosting',p,t,folhas=folhas)
        for alpha in [.03,.1,.3]:
            for l1 in [.3,.8]:add('elasticnet',30,t,alpha=alpha,l1=l1)
        for lags in [[1,2,9,10,11],[1,2,3,9,10,11]]:
            for alpha in [1,10]:add(t=t,lags=lags,alpha=alpha)
    for p in [3,5]:
        for alpha in [10,100]:add('quadratico',p,'raiz',alpha=alpha,alvo='diferenca')
    return lista


def transformar(y,t):
    if t=='raiz':return np.sqrt(y)
    if t=='log1p':return np.log1p(y)
    return np.asarray(y,dtype=float)


def ajustar_prever(hist,c):
    hist=np.asarray(hist,dtype=float)
    z=transformar(hist,c['transformacao'])
    lags=np.asarray(c['lags'],dtype=int)
    inicio=max(30,int(lags.max())+(c['alvo']=='diferenca'))
    indices=np.arange(inicio,len(z))
    if c['janela']:
        indices=indices[-int(c['janela']):]
    if c['alvo']=='diferenca':
        dz=np.diff(z,prepend=np.nan)
        x=dz[indices[:,None]-lags]
        alvo=dz[indices]
        entrada=dz[-lags][None,:]
    else:
        x=z[indices[:,None]-lags]
        alvo=z[indices]
        entrada=z[-lags][None,:]
    if not np.isfinite(x).all():raise ValueError('Defasagem sem histórico válido')
    media=float(alvo.mean());desvio=float(alvo.std()) or 1.0
    ys=(alvo-media)/desvio
    f=c['familia'];alpha=float(c['alpha'])
    if f=='ridge': modelo=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    elif f=='quadratico':modelo=make_pipeline(StandardScaler(),PolynomialFeatures(degree=2,include_bias=False),StandardScaler(),Ridge(alpha=alpha))
    elif f=='spline':modelo=make_pipeline(SplineTransformer(n_knots=3,degree=2,include_bias=False,extrapolation='linear'),StandardScaler(),Ridge(alpha=alpha))
    elif f=='kernel_rbf':modelo=make_pipeline(StandardScaler(),KernelRidge(alpha=alpha,kernel='rbf',gamma=c['gamma']))
    elif f=='svr':modelo=make_pipeline(StandardScaler(),SVR(C=c['C'],gamma=c['gamma'],epsilon=.05))
    elif f.startswith('regime_'):modelo=make_pipeline(Regimes('fase' if f=='regime_fase' else 'nivel'),StandardScaler(),Ridge(alpha=alpha))
    elif f=='elasticnet':modelo=make_pipeline(StandardScaler(),ElasticNet(alpha=alpha,l1_ratio=c['l1'],max_iter=20000,tol=1e-6))
    elif f=='boosting':modelo=make_pipeline(HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=c['folhas'],min_samples_leaf=15,learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=42))
    else:raise ValueError(f)
    parametros={}
    if c['meia_vida']:
        w=2.0**(-(len(z)-1-indices)/c['meia_vida']);w*=len(w)/w.sum()
        parametros[modelo.steps[-1][0]+'__sample_weight']=w
    modelo.fit(x,ys,**parametros)
    previsto=float(modelo.predict(entrada)[0])*desvio+media
    if c['alvo']=='diferenca':previsto+=z[-1]
    if c['transformacao']=='raiz':previsto=max(0,previsto)**2
    elif c['transformacao']=='log1p':previsto=np.expm1(previsto)
    if not np.isfinite(previsto):raise FloatingPointError('Previsão não finita')
    return max(0,previsto),modelo


def prever_bloco(y,inicio,fim,c):
    # Nenhuma previsão é anexada ao histórico. Cada ajuste recebe somente y reais.
    return np.array([ajustar_prever(y[:i],c)[0] for i in range(inicio,fim)])


def avaliar_dev(y,progresso=False,folds=None):
    if folds is None:
        if len(y)!=247:raise ValueError('Passe somente os 247 anos anteriores a 1947.')
        folds=FOLDS
    if folds[-1][1]!=len(y):raise ValueError('O desenvolvimento deve terminar no último bloco.')
    inicio_cv=folds[0][0]
    configs=candidatos();linhas=[];matriz=[]
    inicio_tempo=time.monotonic()
    with threadpool_limits(limits=1):
        for k,c in enumerate(configs):
            previsao=np.concatenate([prever_bloco(y,a,b,c) for a,b in folds])
            matriz.append(previsao)
            erro=previsao-y[inicio_cv:]
            maes=[float(np.mean(abs(erro[a-inicio_cv:b-inicio_cv]))) for a,b in folds]
            linhas.append(dict(id=k,familia=c['familia'],p=len(c['lags']),
                               transformacao=c['transformacao'],alvo=c['alvo'],
                               alpha=c['alpha'],janela=c['janela'],meia_vida=c['meia_vida'],
                               MAE_cv=np.mean(abs(erro)),RMSE_cv=np.sqrt(np.mean(erro**2)),
                               **{f'MAE_fold_{j+1}':v for j,v in enumerate(maes)},
                               config=json.dumps(c,sort_keys=True)))
            if progresso and (k%10==0 or k==len(configs)-1):
                print(f'Candidatos: {k+1}/{len(configs)}; {time.monotonic()-inicio_tempo:.1f}s',flush=True)
    return configs,pd.DataFrame(linhas).sort_values(['MAE_cv','p','id']).reset_index(drop=True),np.array(matriz)


def escolher(y,tabela,matriz):
    # Ensemble pré-definido: média dos três melhores de famílias diferentes.
    # Os integrantes são escolhidos na CV e reescolhidos apenas com blocos passados
    # na avaliação externa. A própria nota CV é otimista e será identificada assim.
    distintos=tabela.sort_values(['MAE_cv','p','id']).drop_duplicates('familia').head(3)
    ids=distintos['id'].astype(int).to_list()
    pred=matriz[ids].mean(axis=0)
    score=float(np.mean(abs(pred-y[120:])))
    melhor=tabela.iloc[0]
    if score<float(melhor['MAE_cv']):
        return {'tipo':'media_tres_familias','ids':ids,'MAE_cv':score}
    return {'tipo':'individual','ids':[int(melhor['id'])],'MAE_cv':float(melhor['MAE_cv'])}
