# Autorregressão de manchas solares

O projeto usa a série anual de manchas solares. Veja `README_manchas_solares.md`
para método, validação e métricas.

```bash
uv sync --locked
uv run --locked python prever_manchas.py
```

O Excel padrão é `resultados_manchas_solares/investigacao_manchas_solares.xlsx`.
A aba `Previsoes` mostra os anos de teste, identificados pelo período no topo.
O modelo padrão é o AR9 transformado, com atualização anual usando dados reais.

Para consultar a investigação adicional:

```bash
uv run --locked python prever_manchas.py --avancado
```
