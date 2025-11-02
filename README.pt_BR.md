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
- Propor uma solução automatizada, replicável e útil à sociedade.[^1]



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

- **Validação dos modelos:** Técnica **Holdout**, separando dados de treino e teste objetivando avaliar o desempenho de cada algoritmo.[^1]


<br><br>
