import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom, poisson

# Carregar dados
df = pd.read_csv("data/campeonato-brasileiro-estatisticas-full.csv")

# Configuração do Streamlit
st.title("Análise de Dados - Campeonato Brasileiro")
st.sidebar.header("Configurações")

# Exibir primeiras linhas
tab1, tab2, tab3 = st.tabs(
    ["📊 Apresentação dos Dados", "📈 Análises Estatísticas", "🎲 Distribuições Probabilísticas"])

with tab1:
    st.header("📌 Apresentação dos Dados")
    st.write("Este conjunto de dados contém estatísticas detalhadas das partidas do Campeonato Brasileiro.")
    st.write("### Primeiras linhas do dataset:")
    st.dataframe(df.head())

    st.write("### Tipos de Variáveis")
    st.write(df.dtypes)

with tab2:
    st.header("📊 Medidas Estatísticas")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_col = st.selectbox(
        "Escolha uma variável numérica para análise:", numeric_cols)

    if selected_col:
        st.write(f"**Média**: {df[selected_col].mean():.2f}")
        st.write(f"**Mediana**: {df[selected_col].median():.2f}")
        st.write(f"**Moda**: {df[selected_col].mode()[0]:.2f}")
        st.write(f"**Desvio Padrão**: {df[selected_col].std():.2f}")
        st.write(f"**Variância**: {df[selected_col].var():.2f}")

        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        ax.set_title(f"Distribuição de {selected_col}")
        st.pyplot(fig)

    st.subheader("Correlação entre variáveis")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab3:
    st.header("🎲 Distribuições Probabilísticas")
    st.write("**Justificativa**:")
    st.write("A distribuição Normal foi escolhida por ser comum em variáveis contínuas como posse de bola.")
    st.write(
        "A distribuição Binomial foi usada pois modela eventos discretos, como gols por partida.")

    dist_col = st.selectbox("Escolha uma variável para modelar:", numeric_cols)

    if dist_col:
        data = df[dist_col].dropna()

        # Distribuição Normal
        mu, sigma = norm.fit(data)
        x = np.linspace(min(data), max(data), 100)
        pdf = norm.pdf(x, mu, sigma)

        fig, ax = plt.subplots()
        sns.histplot(data, kde=True, stat="density", ax=ax)
        ax.plot(x, pdf, 'r-', label="Distribuição Normal")
        ax.set_title(f"Aproximação Normal para {dist_col}")
        ax.legend()
        st.pyplot(fig)

        # Distribuição Binomial
        n = int(data.max())
        p = data.mean() / n
        x_binom = np.arange(0, n)
        binom_pmf = binom.pmf(x_binom, n, p)

        fig, ax = plt.subplots()
        ax.bar(x_binom, binom_pmf, alpha=0.6,
               color='g', label="Distribuição Binomial")
        ax.set_title(f"Distribuição Binomial para {dist_col}")
        ax.legend()
        st.pyplot(fig)
