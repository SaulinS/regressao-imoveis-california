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

Métricas no conjunto de teste (4.128 blocos censitários, 20% dos dados). O alvo está em centenas de milhares de dólares.

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (média) | 0,906 | 1,145 | 0,000 |
| Regressão Linear | 0,533 | 0,746 | 0,576 |
| **Random Forest** | **0,327** | **0,504** | **0,806** |

Validação cruzada com 5 folds nos dados de treino:

| Modelo | RMSE médio | R² médio |
|---|---|---|
| Regressão Linear | 0,721 ± 0,014 | 0,611 ± 0,012 |
| Random Forest | 0,510 ± 0,012 | 0,806 ± 0,006 |

Verificação de overfitting (R² no treino vs no teste):

| Modelo | R² treino | R² teste | Diferença |
|---|---|---|---|
| Regressão Linear | 0,613 | 0,576 | 0,037 |
| Random Forest | 0,974 | 0,806 | 0,168 |

## Principais conclusões

- **Os dois modelos aprenderam com os dados.** Ambos ficaram bem abaixo do erro do baseline, que prevê sempre a média (RMSE de 1,145).
- **O Random Forest foi claramente superior.** Ele reduziu o RMSE em 32,4% em relação à Regressão Linear e explicou cerca de 81% da variação dos preços, contra 58%. Na prática, o erro médio absoluto caiu de cerca de US$ 53 mil para US$ 33 mil por bloco.
- **A diferença não foi sorte da divisão treino/teste.** Na validação cruzada, os dois modelos mantiveram desempenho parecido com o do teste e desvio padrão baixo, o que indica resultados estáveis.
- **O ganho vem da capacidade de modelar relações não lineares.** O preço depende da localização de forma complexa (latitude e longitude combinadas), algo que uma soma ponderada de variáveis não consegue representar.
- **O Random Forest se ajusta bastante aos dados de treino** (R² de 0,974 no treino contra 0,806 no teste). Como o desempenho no teste e na validação cruzada continua alto, o modelo generaliza bem, mas há espaço para regularizar as árvores com `max_depth` ou `min_samples_leaf`.
- **A Regressão Linear continua útil pela interpretabilidade:** seus coeficientes mostram a direção e a intensidade do efeito de cada variável, o que o Random Forest não oferece diretamente.

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
