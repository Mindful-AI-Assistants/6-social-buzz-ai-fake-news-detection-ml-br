 \[**[🇧🇷 Português](README.pt_BR.md)**\] \[[🇺🇸 English](README.md)\]

<br>

# <p align="center"> 6- Social [Buzz AI]() -Detecção de Fake News usando Machine Learning


<br><br>

## 1. Introdução

<br>

- **Fake news** são informações falsas disseminadas principalmente em redes sociais, podendo causar sérios impactos políticos, sociais e de saúde pública.
- O estudo se propôs a aplicar algoritmos de *Machine Learning* (ML) para detectar notícias falsas automaticamente, oferecendo alternativa tecnológica ao combate desse problema.


<br><br>

## 2. Objetivos do Estudo

<br>

- Testar e comparar diferentes algoritmos de ML para detectar fake news.
- Avaliar o desempenho de cada modelo em precisão, sensibilidade e especificidade.
- Propor uma solução automatizada, replicável e útil à sociedade.



<br><br>


## 3. Metodologia Detalhada

<br>

- **Base de dados:** Obtida na plataforma Kaggle, composta por artigos fakes e verdadeiros, ambos com colunas de título, texto, assunto e data.
    - Fake: 23.481 linhas
    - True: 21.417 linhas

<br>

- **Ferramentas utilizadas:** Python e as bibliotecas Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, NLTK.

<br>

- **Processamento dos dados:**
    - Carregamento dos datasets e identificação da coluna alvo (target: 'fake' ou 'true').
    - União das tabelas e embaralhamento dos registros com a biblioteca `Shuffle` para evitar vieses.
    - Remoção das colunas de data e título para focar no texto.
    - Normalização do texto:
        - Transformação para letras minúsculas.
        - Remoção de pontuação (biblioteca `string`).
        - Eliminação dos 'stopwords' (NLTK).
    - Análise visual dos termos mais frequentes por meio de *WordCloud*.

<br>

- **Validação dos modelos:** Técnica **Holdout**, separando dados de treino e teste objetivando avaliar o desempenho de cada algoritmo.


<br><br>


## 4. Algoritmos Aplicados

<br>

Cinco modelos supervisionados foram treinados e avaliados:

<br><br>

| Modelo | Acurácia | Observações |
| :-- | :-- | :-- |
| Regressão Logística | 98,92% | Alta precisão, matriz de confusão mostra baixo índice de erro. |
| Decision Tree | 99,6% | Melhor desempenho geral e menor erro nos resultados. |
| Random Forest | 98,74% | Bom desempenho, matriz de confusão consistente. |
| Support Vector Machine | 99,5% | Ótima acurácia e precisão, modelo robusto para textos. |
| K-Nearest Neighbors (KNN) | 60,84% | Baixo desempenho, elevado número de falsos negativos. |

<br><br>

**Métricas computadas via matriz de confusão** (incluindo VP, VN, FP, FN) e valores de precisão e sensibilidade detalhados para cada modelo.

#### Exemplo: Matriz de confusão simplificada (Decision Tree)

- Verdadeiros Positivos (VP): 4.711
- Falsos Positivos (FP): 13
- Falsos Negativos (FN): 22
- Verdadeiros Negativos (VN): 4.234

<br><br>


## 5. Métricas de Avaliação

- **Acurácia:** Percentual total de acertos do modelo.
- **Precisão:** Quão corretamente detecta fake news.
- **Sensibilidade:** Capacidade de identificar corretamente as verdadeiras fake news.
- **Especificidade:** Capacidade de acertar os valores negativos (verdadeiras).

<br>

#### Desempenho dos Modelos

<br><br>

| Modelo | Precisão | Sensibilidade | Especificidade |
| :-- | :-- | :-- | :-- |
| Regressão Logística | 98% | 99% | 98% |
| Decision Tree | 98,5% | 99% | 99% |
| Random Forest | 98,5% | 99% | 98% |
| SVM | 99% | 99% | 99% |
| KNN | 99% | 57% | 19% |

<br><br>


## 6. Resultados Detalhados

<br>

- Quatro dos cinco modelos atingiram acurácia superior a 90%.
- O KNN obteve desempenho insatisfatório, principalmente devido ao alto número de **falsos negativos** (sensibilidade de 57%).
- Decision Tree e SVM destacaram-se como os mais eficientes.
- O processo de tratamento de dados e seleção de features foi fundamental para o sucesso dos modelos.


<br><br>

## 7. Limitações e Propostas Futuras

<br>

- **Limitações do trabalho:** Dificuldade para encontrar base de dados padronizada (especialmente em português), poucos sistemas aplicados à realidade brasileira.
- **Perspectivas:** Avaliar novos algoritmos (Naive Bayes, Boosting, K-means, Gradiente Descendente), aplicar outras técnicas de validação, ampliar bases em português, incluir validação cruzada (K-fold, Leave-one-out) e desenvolver aplicações web para uso público


<br><br>

