# Previsão de Preços de Imóveis: Regressão Linear vs Random Forest

Projeto de aprendizado de máquina supervisionado que compara um modelo linear com um modelo baseado em árvores na previsão do valor mediano de imóveis na Califórnia.

## Objetivo

Avaliar se o ganho de desempenho do Random Forest Regressor, que modela relações não lineares, compensa a perda de interpretabilidade em relação à Regressão Linear.

## Dataset

**California Housing** (censo americano de 1990), disponível no scikit-learn.

- 20.640 registros, cada um representando um bloco censitário
- 8 variáveis: renda mediana, idade das casas, média de cômodos e quartos, população, ocupação média, latitude e longitude
- Alvo: valor mediano das casas, em centenas de milhares de dólares

## Metodologia

1. Análise exploratória: distribuição do alvo, correlações e distribuição geográfica dos preços
2. Divisão treino/teste (80/20) com semente fixa
3. Baseline que prevê sempre a média, como referência mínima
4. Regressão Linear em pipeline com padronização (`StandardScaler`), evitando vazamento de dados
5. Random Forest Regressor com 200 árvores
6. Validação cruzada com 5 folds para verificar a estabilidade dos resultados
7. Comparação por MAE, RMSE e R², análise de resíduos e verificação de overfitting

## Resultados

<!-- Preencha com os valores obtidos ao executar o notebook -->

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (média) | | | |
| Regressão Linear | | | |
| Random Forest | | | |

## Principais conclusões

<!-- Escreva com suas palavras o que os resultados mostraram -->

## Tecnologias

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

## Como executar

```bash
git clone https://github.com/SaulinS/regressao-imoveis-california.git
cd regressao-imoveis-california
pip install -r requirements.txt
jupyter notebook regressao_linear_vs_random_forest.ipynb
```

Na primeira execução, o scikit-learn baixa o dataset automaticamente (é preciso estar conectado à internet).

## Próximos passos

- Ajuste de hiperparâmetros com `GridSearchCV`
- Modelos de gradient boosting (XGBoost, LightGBM)
- Interpretação das previsões com SHAP

## Autor

**Saulo Sousa Cunha** · Engenharia de Computação, Universidade SENAI CIMATEC
[LinkedIn](https://linkedin.com/in/saulo-sousa-b2187b256) · [GitHub](https://github.com/SaulinS)
