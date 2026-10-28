"""Investigação temporal e comparação retrospectiva de modelos autorregressivos."""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from openpyxl.styles import Alignment, Font, PatternFill

from manchas_solares import carregar, executar as executar_original, erro_relativo
from manchas_modelos import selecionar, prever_bloco, ajustar_prever

FOLDS = [(120, 152), (152, 184), (184, 216), (216, 247)]


def metricas(real, previsto, escala):
    real, previsto = np.asarray(real), np.asarray(previsto)
    erro = previsto - real
    relativos = erro_relativo(real, previsto)
    return dict(MAE=np.mean(abs(erro)), RMSE=np.sqrt(np.mean(erro**2)),
                R2=r2_score(real, previsto), MAPE=np.nanmean(relativos),
                WAPE=np.sum(abs(erro))/np.sum(abs(real)),
                MASE=np.mean(abs(erro))/escala, Vies=np.mean(erro))


def avaliar_aninhado(tabela, predicoes, y):
    # O primeiro bloco serve à escolha inicial. Cada bloco posterior é avaliado
    # com o candidato escolhido exclusivamente nos blocos anteriores.
    linhas, detalhes = [], []
    for fold_atual in [2, 3, 4]:
        n_anteriores = fold_atual - 1
        colunas = [f'MAE_fold_{f}' for f in range(1, fold_atual)]
        # Os blocos anteriores têm 32 anos cada, portanto sua média é exata.
        notas = tabela[colunas].mean(axis=1)
        ranking = tabela.assign(nota_anterior=notas).sort_values(['nota_anterior', 'p', 'alpha'])
        vencedor = ranking.iloc[0]
        previsoes = pd.DataFrame(predicoes[int(vencedor['candidato'])])
        previsoes = previsoes[previsoes['fold'].eq(fold_atual)].copy()
        previsoes['ano'] = previsoes['indice'] + 1700
        previsoes['persistencia'] = y[previsoes['indice'].to_numpy()-1]
        previsoes['candidato'] = int(vencedor['candidato'])
        linhas.append(previsoes)
        detalhes.append({
            'Bloco externo': fold_atual,
            'Período': f"{previsoes['ano'].min()}–{previsoes['ano'].max()}",
            'Blocos anteriores usados na escolha': n_anteriores,
            'Transformação': vencedor['transformacao'], 'p': int(vencedor['p']),
            'Alpha': vencedor['alpha'], 'Atualização': vencedor['atualizacao'],
            'MAE externo': np.mean(abs(previsoes['predito']-previsoes['real'])),
            'MAE persistência': np.mean(abs(previsoes['persistencia']-previsoes['real'])),
        })
    return pd.concat(linhas, ignore_index=True), pd.DataFrame(detalhes)


def bootstrap_ganho(real, anterior, novo, bloco=11, repeticoes=5000):
    ganho = abs(anterior-real) - abs(novo-real)
    rng = np.random.default_rng(42)
    n = len(ganho)
    inicios = rng.integers(0, n-bloco+1, size=(repeticoes, int(np.ceil(n/bloco))))
    indices = (inicios[:, :, None]+np.arange(bloco)).reshape(repeticoes, -1)[:, :n]
    amostras = ganho[indices].mean(axis=1)
    return pd.DataFrame([{
        'Ganho médio MAE': ganho.mean(), 'Limite 2,5%': np.quantile(amostras, .025),
        'Limite 97,5%': np.quantile(amostras, .975), 'Comprimento bloco': bloco,
        'Reamostragens': repeticoes,
    }])


def coeficientes(modelo, p):
    escala = modelo.named_steps['standardscaler']
    reg = modelo.steps[-1][1]
    beta = reg.coef_ / escala.scale_
    intercepto = reg.intercept_ - np.dot(beta, escala.mean_)
    return pd.DataFrame({'Parâmetro': ['Intercepto']+[f'Lag {i}' for i in range(1,p+1)],
                         'Valor na escala transformada': [intercepto]+list(beta)})


def investigar(pasta, progresso=False):
    dados, fonte = carregar(pasta)
    y = dados['Atividade solar'].to_numpy()
    # A busca recebe somente o desenvolvimento. O teste não entra nesta função.
    escolhido, busca, oof, predicoes = selecionar(y[:247], progresso=progresso)
    cv_aninhado, blocos_aninhados = avaliar_aninhado(busca, predicoes, y[:247])
    config_referencia = dict(transformacao='original', p=19, alpha=0, atualizacao='fixo')
    referencia_oof = np.concatenate([prever_bloco(y[:247], a, b, config_referencia) for a,b in FOLDS])
    comparacao_cv = []
    escala_dev = np.mean(abs(np.diff(y[:120])))
    for nome, real, previsto in [
        ('Selecionado na própria CV (otimista)', y[120:247], oof['predito'].to_numpy()),
        ('AR(19) pré-fixado para referência', y[120:247], referencia_oof),
        ('Persistência nos quatro blocos', y[120:247], y[119:246]),
        ('Procedimento aninhado, três blocos externos', cv_aninhado['real'].to_numpy(), cv_aninhado['predito'].to_numpy()),
        ('Persistência nos mesmos blocos externos', cv_aninhado['real'].to_numpy(), cv_aninhado['persistencia'].to_numpy()),
    ]:
        comparacao_cv.append({'Avaliação': nome, 'Anos':len(real), **metricas(real, previsto, escala_dev)})

    # A escolha está congelada. Só agora calculamos a comparação já vista de 1947–2008.
    original = executar_original(pasta)
    real = y[247:]
    anterior = original['teste']['Valor predito'].to_numpy()
    novo = prever_bloco(y, 247, len(y), escolhido)
    persistencia = y[246:-1]
    escala = np.mean(abs(np.diff(y[:247])))
    comparacao = pd.DataFrame([
        {'Modelo':nome, 'Anos':len(real), **metricas(real, previsto, escala)}
        for nome, previsto in [('Anterior AR(19)', anterior), ('Melhorado', novo), ('Persistência', persistencia)]
    ])
    # Intervalo empírico calibrado só nos erros externos do procedimento aninhado.
    # É um intervalo descritivo, sem garantia de cobertura temporal.
    erros_calibracao = abs(cv_aninhado['predito']-cv_aninhado['real']).to_numpy()
    raio = np.quantile(erros_calibracao, min(1, np.ceil((len(erros_calibracao)+1)*.90)/len(erros_calibracao)), method='higher')
    resultados = pd.DataFrame({
        'Ano de origem':dados['Ano'].iloc[246:-1].to_numpy(),
        'Ano previsto':dados['Ano'].iloc[247:].to_numpy(),
        'Valor real':real, 'Valor predito':novo,
        'Taxa de erro (%)':erro_relativo(real,novo), 'Erro absoluto':abs(novo-real),
        'Predito anterior AR19':anterior, 'Erro anterior':abs(anterior-real),
        'Persistência':persistencia, 'Limite inferior empírico':np.maximum(0,novo-raio),
        'Limite superior empírico':novo+raio,
    })
    resultados['Dentro do intervalo'] = (real >= resultados['Limite inferior empírico']) & (real <= resultados['Limite superior empírico'])
    faixas = pd.cut(real, bins=[-np.inf,10,50,100,np.inf],labels=['Até 10','10 a 50','50 a 100','Acima de 100'])
    diagnosticos_faixa = []
    for nome, previsto in [('Anterior AR19',anterior),('Melhorado',novo)]:
        for faixa in faixas.categories:
            mask = np.asarray(faixas == faixa)
            diagnosticos_faixa.append({'Modelo':nome,'Faixa real':faixa,'Anos':int(mask.sum()),**metricas(real[mask],previsto[mask],escala)})
    periodos=[]
    for a,b in [(1947,1967),(1968,1988),(1989,2008)]:
        mask = resultados['Ano previsto'].between(a,b).to_numpy()
        for nome,previsto in [('Anterior AR19',anterior),('Melhorado',novo)]:
            periodos.append({'Período':f'{a}–{b}','Modelo':nome,**metricas(real[mask],previsto[mask],escala)})

    residuos=[]
    for nome,erros in [('Anterior AR19',anterior-real),('Melhorado',novo-real)]:
        lb=acorr_ljungbox(erros,lags=[5,10,20])
        for lag,row in lb.iterrows():
            residuos.append({'Modelo':nome,'Defasagem':lag,'Ljung-Box':row['lb_stat'],'p-valor':row['lb_pvalue']})
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        adf=adfuller(y[:247],autolag='AIC')
        kp=kpss(y[:247],regression='c',nlags='auto')
    testes = pd.DataFrame([
        {'Teste':'ADF (nível, desenvolvimento)','Estatística':adf[0],'p-valor':adf[1],
         'Leitura':'H0: raiz unitária; p<0,05 rejeita H0 nesta especificação'},
        {'Teste':'KPSS (nível, desenvolvimento)','Estatística':kp[0],'p-valor':kp[1],
         'Leitura':'H0: estacionariedade em nível; p informado limitado pela tabela (>=0,10) nesta série'},
    ])
    # Modelo após observar até 2008: os coeficientes deste ajuste serviriam para 2009.
    _,ajuste_atual=ajustar_prever(y,escolhido)
    correlacoes = pd.DataFrame([
        {'Modelo':nome,'Correlação Spearman entre |erro| e real':spearmanr(abs(pred-real),real).statistic}
        for nome,pred in [('Anterior AR19',anterior),('Melhorado',novo)]
    ])
    config = pd.DataFrame([
        ['Seleção',json.dumps(escolhido,ensure_ascii=False)],
        ['Candidatos',len(busca)],
        ['Desenvolvimento','1700–1946; quatro blocos de avaliação: 1820–1851, 1852–1883, 1884–1915, 1916–1946'],
        ['Busca','p em 2,3,5,9,11,15,19; alpha em 0,1,10; original/raiz/log1p; fixo/expansivo'],
        ['Padronização','Média e desvio calculados só com as entradas do ajuste de cada previsão'],
        ['AR transformado','AR linear na raiz quadrada; retorno à escala original elevando ao quadrado, após limitar a zero'],
        ['Atualização','Coeficientes reajustados a cada ano usando somente os anos já observados; p e alpha congelados'],
        ['Hold-back','30 em todos os modelos da busca; mesma sequência de alvos por ajuste'],
        ['Escolha','Menor MAE médio nos quatro blocos; desempate por menor p e alpha'],
        ['Validação aninhada','Cada bloco externo 2 a 4 escolhe o candidato pelos erros dos blocos anteriores; não usa seu próprio alvo na escolha'],
        ['Comparação retrospectiva','1947–2008 já foi visto no experimento anterior. Não constitui teste novo nem prova de generalização'],
        ['Referência anterior','AR19 com coeficientes fixos ajustados até 1946, reproduzido sem alterações'],
        ['MAPE e WAPE','MAPE exclui real zero; WAPE divide soma dos erros absolutos pela soma dos reais'],
        ['Intervalo empírico',f'Raio {raio:.4f}, calibrado a 90% com erros externos aninhados; sem garantia formal de cobertura'],
        ['Cobertura retrospectiva',float(resultados['Dentro do intervalo'].mean())],
        ['Fonte',fonte['url_csv']],
        ['Ljung-Box','Nos erros de previsão, model_df=0; diagnóstico descritivo, sem ajuste por múltiplos testes'],
        ['Limite','126 candidatos podem sobreajustar a seleção. O resultado da própria busca é otimista; consultar a avaliação aninhada'],
    ],columns=['Item','Valor'])
    return dict(escolhido=escolhido, busca=busca, oof=oof,
                cv_aninhado=cv_aninhado, blocos_aninhados=blocos_aninhados,
                comparacao_cv=pd.DataFrame(comparacao_cv), comparacao=comparacao,
                resultados=resultados, faixas=pd.DataFrame(diagnosticos_faixa),
                periodos=pd.DataFrame(periodos), residuos=pd.DataFrame(residuos),
                estacionariedade=testes, correlacoes=correlacoes,
                bootstrap=bootstrap_ganho(real,anterior,novo),
                coeficientes=coeficientes(ajuste_atual,escolhido['p']),
                configuracao=config, dados=dados, original=original,
                raio=raio, avisos_estacionariedade=[str(w.message) for w in avisos])


def exportar(e,pasta):
    destino=Path(pasta)/'resultados_manchas_solares'
    destino.mkdir(exist_ok=True)
    saida=destino/'investigacao_manchas_solares.xlsx'
    r = e['resultados']
    previsoes = pd.DataFrame({
        'Ano de origem': r['Ano de origem'],
        'Ano previsto': r['Ano previsto'],
        'Valor real': r['Valor real'],
        'Predito AR(9)': r['Valor predito'],
        'Erro absoluto AR(9)': r['Erro absoluto'],
        'Erro AR(9) (%)': r['Taxa de erro (%)'],
        'Predito AR(19)': r['Predito anterior AR19'],
        'Erro absoluto AR(19)': r['Erro anterior'],
        'Erro AR(19) (%)': erro_relativo(r['Valor real'], r['Predito anterior AR19']),
    })
    metricas = e['comparacao'].loc[
        e['comparacao']['Modelo'].isin(['Melhorado', 'Anterior AR(19)']),
        ['Modelo', 'Anos', 'MAE', 'RMSE', 'R2', 'MAPE']
    ].copy()
    metricas['Modelo'] = metricas['Modelo'].replace({
        'Melhorado': 'AR(9)', 'Anterior AR(19)': 'AR(19) — referência'
    })
    metricas = metricas.sort_values('Modelo').rename(columns={'R2': 'R²', 'MAPE': 'MAPE (%)'})
    metodologia = pd.DataFrame([
        ['Dados', 'Atividade solar anual; 309 registros, de 1700 a 2008'],
        ['Período apresentado', '1947 a 2008; 62 previsões de um ano à frente'],
        ['Treino inicial', '1700 a 1884'],
        ['Validação e escolha do modelo', 'Desenvolvimento até 1946; seleção temporal antes de avaliar 1947–2008'],
        ['AR(9)', 'Últimos 9 valores reais; raiz quadrada, Ridge alpha=1 e coeficientes reajustados anualmente'],
        ['AR(19) de referência', 'Últimos 19 valores reais; escala original e coeficientes fixos ajustados até 1946'],
        ['Diferença entre os modelos', 'Além das defasagens, diferem a transformação, a regularização e a atualização dos coeficientes'],
        ['Entradas', 'Somente observações reais anteriores ao ano previsto; previsões nunca são usadas como entrada'],
        ['Erro percentual', 'abs(predito - real) / abs(real), exibido como porcentagem; real zero fica vazio'],
        ['Métricas', 'MAE e RMSE em unidades do índice; menores são melhores. R² não é percentual de acerto'],
        ['Limite da comparação', '1947–2008 já foi consultado durante a investigação: avaliação retrospectiva'],
        ['Fonte', e['fonte']['url_csv'] if 'fonte' in e else e['original']['fonte']['url_csv']],
    ], columns=['Item', 'Descrição'])
    tabelas = {'Previsoes': previsoes, 'Metricas': metricas, 'Metodologia': metodologia}
    with pd.ExcelWriter(saida,engine='openpyxl') as writer:
        for nome,tabela in tabelas.items():
            linha_inicio = 2 if nome == 'Previsoes' else 0
            tabela.to_excel(writer,sheet_name=nome,index=False,startrow=linha_inicio)
            sheet=writer.sheets[nome]
            linha_cabecalho = linha_inicio + 1
            sheet.freeze_panes=f'A{linha_cabecalho + 1}'
            sheet.auto_filter.ref=f'A{linha_cabecalho}:{sheet.cell(sheet.max_row, sheet.max_column).coordinate}'
            if nome == 'Previsoes':
                inicio = int(tabela['Ano previsto'].min())
                fim = int(tabela['Ano previsto'].max())
                sheet['A1'] = f'Dados de {inicio} até {fim}'
                sheet['A1'].font = Font(name='Arial',size=14,bold=True,color='243B53')
                sheet.row_dimensions[1].height = 27
                sheet['A2'] = 'Previsões de teste — comparação retrospectiva; um ano à frente'
                sheet['A2'].font = Font(name='Arial',size=10,italic=True,color='555555')
            sheet.sheet_view.showGridLines=False
            sheet.row_dimensions[linha_cabecalho].height=32
            for celula in sheet[linha_cabecalho]:
                celula.fill=PatternFill('solid',fgColor='243B53')
                celula.font=Font(name='Arial',size=10,bold=True,color='FFFFFF')
                celula.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
                sheet.column_dimensions[celula.column_letter].width=min(42,max(21,len(str(celula.value))+3))
            for linha in sheet.iter_rows(min_row=linha_cabecalho + 1):
                for celula in linha:
                    celula.font=Font(name='Arial',size=10)
                    celula.alignment=Alignment(vertical='center')
                    if isinstance(celula.value,float):celula.number_format='0.000'
                    cabecalho=tabela.columns[celula.column-1]
                    if cabecalho in ['Erro AR(9) (%)','Erro AR(19) (%)','MAPE (%)']:
                        celula.number_format='0.00%'
            if nome=='Metodologia':
                sheet.column_dimensions['A'].width=31
                sheet.column_dimensions['B'].width=115
                for linha in sheet.iter_rows(min_row=linha_cabecalho + 1):
                    linha[1].alignment=Alignment(wrap_text=True,vertical='center')
                    sheet.row_dimensions[linha[0].row].height=46
            elif nome=='Validacao_temporal':
                sheet.column_dimensions['A'].width=61
            elif nome=='Estacionariedade':
                sheet.column_dimensions['A'].width=42
                sheet.column_dimensions['D'].width=100
            elif nome=='Correlacao_erros':
                sheet.column_dimensions['B'].width=58
    (destino/'modelo_selecionado.json').write_text(json.dumps(e['escolhido'],ensure_ascii=False,indent=2)+'\n')
    return saida


def graficos(e,pasta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    destino=Path(pasta)/'resultados_manchas_solares'
    r=e['resultados'];anos=r['Ano previsto']
    fig,axes=plt.subplots(2,1,figsize=(13,8))
    axes[0].fill_between(anos,r['Limite inferior empírico'],r['Limite superior empírico'],color='#BA6F30',alpha=.12,label='Intervalo empírico de referência 90%')
    axes[0].plot(anos,r['Valor real'],label='Real',color='#243B53')
    axes[0].plot(anos,r['Predito anterior AR19'],label='Anterior AR19',color='#999999',alpha=.8)
    axes[0].plot(anos,r['Valor predito'],label='Melhorado: AR9 na raiz + Ridge + atualização anual',color='#BA6F30')
    axes[0].set(title='Comparação retrospectiva — 1947–2008 já observado',ylabel='Atividade solar')
    axes[0].legend(fontsize=9)
    axes[1].axhline(0,color='#999999',linewidth=.8)
    axes[1].plot(anos,r['Erro anterior']-r['Erro absoluto'],marker='.',color='#45866C')
    axes[1].set(title='Redução do erro absoluto por ano (positivo favorece o melhorado)',xlabel='Ano',ylabel='Erro anterior − erro novo')
    for ax in axes:ax.grid(True,alpha=.2)
    fig.tight_layout();fig.savefig(destino/'comparacao_modelos.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for coluna,nome,cor in [('Predito anterior AR19','Anterior','#999999'),('Valor predito','Melhorado','#BA6F30')]:
        erro=r[coluna]-r['Valor real']
        axes[0].scatter(r['Valor real'],abs(erro),label=nome,alpha=.6,s=20,color=cor)
    axes[0].set(xlabel='Valor real',ylabel='Erro absoluto',title='Erro por nível de atividade');axes[0].legend()
    plot_acf(r['Valor predito']-r['Valor real'],lags=20,ax=axes[1],title='ACF dos erros do melhorado')
    blocks=e['blocos_aninhados'];x=np.arange(len(blocks))
    axes[2].bar(x-.18,blocks['MAE externo'],width=.36,label='Procedimento aninhado',color='#BA6F30')
    axes[2].bar(x+.18,blocks['MAE persistência'],width=.36,label='Persistência',color='#999999')
    axes[2].set(xticks=x,xticklabels=blocks['Período'],ylabel='MAE',title='Validação externa por período')
    axes[2].tick_params(axis='x',rotation=25);axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(destino/'diagnostico_erros.png',dpi=160);plt.close(fig)


def relatorio(e,pasta):
    m=e['comparacao'].set_index('Modelo');a=m.loc['Anterior AR(19)'];n=m.loc['Melhorado']
    boot=e['bootstrap'].iloc[0]
    cv=e['comparacao_cv'].set_index('Avaliação')
    nested=cv.loc['Procedimento aninhado, três blocos externos']
    naive=cv.loc['Persistência nos mesmos blocos externos']
    faixas=e['faixas']
    linhas_faixas=[]
    for _,r in faixas.iterrows():
        linhas_faixas.append(f"| {r['Modelo']} | {r['Faixa real']} | {r['Anos']} | {r['MAE']:.2f} | {r['MAPE']:.1%} |")
    t=e['estacionariedade'].set_index('Teste')
    linhas=[
        '# Investigação do modelo de manchas solares',
        '',
        'O modelo melhorou nesta comparação histórica, mas o período de 1947–2008 já foi visto. Não é um teste novo.',
        '',
        '## O que foi investigado',
        '',
        '- Dependência temporal e estacionariedade no desenvolvimento anterior a 1947.',
        '- Erros por faixa de atividade, por período e autocorrelação dos erros de previsão.',
        '- 126 configurações: 7 quantidades de defasagens, 3 transformações, 3 regularizações e 2 políticas de atualização.',
        '- Quatro blocos cronológicos de validação e uma avaliação aninhada em que cada escolha usa só blocos anteriores.',
        '',
        '## Modelo escolhido sem usar os resultados de 1947–2008 na busca',
        '',
        f"Configuração: {json.dumps(e['escolhido'],ensure_ascii=False)}.",
        'É uma autorregressão linear na raiz quadrada da atividade, com Ridge para reduzir a instabilidade dos coeficientes. A volta à escala original eleva a previsão ao quadrado. Não foram adicionadas variáveis externas.',
        'Os coeficientes são reajustados a cada ano, apenas com o passado. A ordem, a transformação e a regularização permanecem congeladas depois da seleção.',
        'Todas as padronizações são ajustadas só no histórico disponível. Não há divisão aleatória nem previsão usando o próprio valor alvo.',
        '',
        '## Comparação retrospectiva em 62 anos',
        '',
        '| Modelo | MAE | RMSE | R² | MAPE | WAPE |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for nome,r in m.iterrows():
        linhas.append(f"| {nome} | {r['MAE']:.2f} | {r['RMSE']:.2f} | {r['R2']:.3f} | {r['MAPE']:.2%} | {r['WAPE']:.2%} |")
    linhas += [
        '',
        f"Redução frente ao anterior: MAE {(1-n['MAE']/a['MAE']):.1%}; RMSE {(1-n['RMSE']/a['RMSE']):.1%}.",
        f"Bootstrap pareado por blocos de 11 anos, 5.000 reamostragens: ganho médio de MAE {boot['Ganho médio MAE']:.2f}; intervalo percentil de 95% [{boot['Limite 2,5%']:.2f}, {boot['Limite 97,5%']:.2f}]. É uma análise descritiva, com poucos ciclos e sem ajuste pela seleção do modelo.",
        '',
        '## Validação em períodos diferentes',
        '',
        f"O procedimento aninhado, avaliado em {int(nested['Anos'])} anos externos, obteve MAE {nested['MAE']:.2f}, contra {naive['MAE']:.2f} da persistência nos mesmos anos.",
        'Cada bloco externo escolheu sua configuração pelos blocos anteriores. Isso avalia o procedimento de seleção; não equivale a avaliar a configuração final nos mesmos dados usados para escolhê-la.',
        'O MAE do melhor candidato nos quatro blocos é uma medida usada na seleção e, por isso, otimista. O Excel mostra separadamente esse valor e a avaliação aninhada.',
        '',
        '## Por que o percentual de erro permanece alto',
        '',
        '| Modelo | Faixa real | Anos | MAE | MAPE |',
        '| --- | --- | ---: | ---: | ---: |',
        *linhas_faixas,
        '',
        'A base tem mínimos próximos de zero. Neles, um erro absoluto pequeno pode virar um percentual enorme. A transformação busca estabilizar a amplitude, mas não elimina a dificuldade de antecipar picos e mínimos. Não foram excluídos mínimos para melhorar artificialmente os resultados.',
        '',
        '## Diagnósticos e limites',
        '',
        f"No desenvolvimento, ADF apresentou p={t.iloc[0]['p-valor']:.4f}; KPSS retornou p na borda superior da tabela (>=0,10). Os resultados não indicaram necessidade clara de diferenciação nesta especificação; não provam estacionariedade definitiva.",
        'Ljung–Box dos erros de previsão, por defasagem, está no Excel. É descritivo: não houve correção por múltiplos testes, e rejeição isolada não demonstra um padrão fácil de prever.',
        f"O intervalo empírico calibrado em erros externos anteriores a 1947 cobriu {e['resultados']['Dentro do intervalo'].mean():.1%} dos anos da comparação retrospectiva. A cobertura nominal de referência é 90%, sem garantia formal para esta série temporal.",
        'Não aumentamos a busca depois de observar o desempenho final da configuração escolhida. Mesmo assim, toda esta investigação foi motivada por um teste já consultado: é necessário avaliar em anos ainda não utilizados ou em outra série independente antes de afirmar generalização.',
        'Os dados disponíveis terminam em 2008. Não foram baixados anos adicionais nem misturadas versões do índice de atividade solar para criar um teste aparentemente novo.',
        '',
        '## Executar',
        '',
        '```bash',
        'uv sync --locked',
        'uv run --locked python prever_manchas.py',
        '```',
        '',
        'O script refaz busca, diagnósticos, comparação, planilha e gráficos. A versão anterior pode ser reproduzida com `uv run --locked python prever_manchas_original.py`.',
        '',
        '## Referências',
        '',
        '- Dataset: https://www.statsmodels.org/stable/datasets/generated/sunspots.html',
        '- Autorregressões: https://www.statsmodels.org/stable/examples/notebooks/generated/autoregressions.html',
        '- Diagnóstico Ljung–Box: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html',
    ]
    destino=Path(pasta)/'resultados_manchas_solares/relatorio_investigacao.md'
    destino.write_text('\n'.join(linhas)+'\n')
    return destino
