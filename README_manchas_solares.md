# Manchas solares — investigação e melhoria da autorregressão

```bash
cd /Users/ryanbernardo/Documents/Faculdade/MachineLearning/projeto_auto_regressao
uv sync --locked
uv run --locked python prever_manchas.py
```

O comando refaz a investigação completa e gera o Excel, os gráficos e o relatório
em `resultados_manchas_solares`. O arquivo principal é `investigacao_manchas_solares.xlsx`.

Dataset: 309 registros anuais (1700–2008), já salvo em `dados_manchas_solares`.
O SHA256 e a fonte estão em `fonte.json`. Não é necessário baixar novamente.

O experimento compara 126 configurações exclusivamente antes de 1947:
sete ordens AR (2, 3, 5, 9, 11, 15, 19), três transformações (original, raiz e
log1p), três regularizações (OLS, Ridge 1 e Ridge 10) e duas atualizações
(coeficientes fixos por bloco ou reajuste anual com histórico expansivo).
Os quatro blocos de validação são 1820–1851, 1852–1883, 1884–1915 e 1916–1946.
Cada previsão é de um ano à frente e usa apenas anos reais já observados.

A transformação e a padronização são calculadas sem futuro. Todos os candidatos
usam hold-back de 30 anos. O menor MAE médio determina a configuração final.
Na validação aninhada, cada bloco externo escolhe a configuração só pelos
blocos anteriores. Os 95 anos externos avaliam o procedimento de seleção.

O modelo selecionado é autorregressivo na escala transformada. Não usa features
externas. Com raiz quadrada, a previsão não negativa volta à escala original
pelo quadrado. Ridge regulariza os coeficientes; as entradas são padronizadas.

Atenção ao protocolo: 1947–2008 já foi analisado no experimento original.
O novo resultado nesse período é uma comparação retrospectiva, não um teste
inédito. A busca não consulta esse período, mas foi motivada por resultados
que já conhecemos. Leia os limites e a incerteza em `relatorio_investigacao.md`.

O Excel inclui previsões reais/preditas, erro percentual, comparação com AR19 e
persistência, validação, busca, erros por faixa e período, resíduos, testes de
estacionariedade, bootstrap por blocos e intervalos empíricos. O MAPE ignora
real zero e pode crescer perto de zero. Não se deve interpretar R² como acurácia.

Arquivos de código:

- `manchas_modelos.py`: candidatos, transformação, ajuste e validação temporal.
- `investigacao_manchas.py`: investigação, comparação e geração de saídas.
- `prever_manchas.py`: comando principal.
- `manchas_solares.py`: referência AR19 original e carregamento da base.

Para reproduzir o experimento anterior:

```bash
uv run --locked python prever_manchas_original.py
```

Para usar o notebook:

```bash
uv run --locked --extra notebook jupyter lab
```

Abra `manchas_solares.ipynb`. O notebook anterior está preservado em
`manchas_solares_original.ipynb`.

## Investigação avançada adicional

```bash
uv run --locked python prever_manchas.py --avancado
uv run --locked python prever_manchas.py --avancado --refazer-busca
```

São mais 177 configurações, incluindo autorregressões não lineares. O candidato
escolhido na validação melhorou o erro percentual, mas piorou MAE e RMSE no período
retrospectivo. Portanto, o AR9 permanece como padrão. Os resultados adicionais estão
em `investigacao_avancada_manchas.xlsx` e `relatorio_avancado.md`.
Todos os candidatos usam exclusivamente dados reais anteriores como entrada.
O cache registra hashes do código e do CSV; pode ser refeito com a opção acima.

## Planilha para análise

O Excel padrão contém apenas `Previsoes`, `Metricas` e `Metodologia`.
Compara AR(9) e AR(19), com valores reais, preditos e erros. Intervalos empíricos,
bootstrap e abas de busca/diagnóstico ficam fora dessa planilha. O código e os
relatórios técnicos preservam a investigação. Os modelos também diferem em
transformação, regularização e política de atualização, além da ordem AR.
