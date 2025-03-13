import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom, poisson

df = pd.read_excel("data/campeonato-brasileiro-estatisticas-full.xlsx")

st.title("Análise de Dados - Campeonato Brasileiro")
st.sidebar.header("Configurações")

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Apresentação dos Dados", "📈 Análises Estatísticas", "🎲 Distribuições Probabilísticas", "📈 Análise Crítica"])

with tab1:
    st.header("📌 Apresentação dos Dados")
    st.write("Este conjunto de dados contém estatísticas detalhadas das partidas do Campeonato Brasileiro.")
    st.write("### Primeiras linhas do dataset:")
    st.dataframe(df.head())

    st.write("### Classificação das Variáveis:")
    variaveis = {
    "Variável": ["clube", "chutes", "chutes_no_alvo", "posse_de_bola", "passes", 
                 "faltas", "cartao_amarelo", "cartao_vermelho", "conversao"],
    "Tipo": ["Qualitativa", "Quantitativa", "Quantitativa", "Quantitativa", "Quantitativa",
             "Quantitativa", "Quantitativa", "Quantitativa", "Quantitativa"],
    "Subtipo": ["Nominal", "Discreta", "Discreta", "Contínua", "Discreta",
                "Discreta", "Discreta", "Discreta", "Contínua"]
    }

    df_variaveis = pd.DataFrame(variaveis)
    st.dataframe(df_variaveis)

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

with tab4:
    st.header("Eficiência Ofensiva")
    
    st.subheader("Relação entre Chutes e Chutes no Alvo")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='chutes', y='chutes_no_alvo', ax=ax)
    ax.set_title("Chutes vs. Chutes no Alvo")
    st.pyplot(fig)
    
    st.write("Observa-se uma forte correlação positiva: partidas com mais chutes tendem a ter mais chutes no alvo, "
             "mas a taxa de conversão pode variar.")

    st.header("Controle do Jogo")
    
    st.subheader("Posse de Bola vs. Passes")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='posse_de_bola', y='passes', ax=ax)
    ax.set_title("Posse de Bola vs. Número de Passes")
    st.pyplot(fig)
    
    st.write("Ou seja, times com maior posse de bola tendem a realizar mais passes, evidenciando o controle do jogo.")
    
    st.subheader("Posse de Bola vs. Chutes")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x='posse_de_bola', y='chutes', ax=ax2)
    ax2.set_title("Posse de Bola vs. Chutes")
    st.pyplot(fig2)
    
    st.write("Embora o domínio de bola facilite a criação de jogadas, nem sempre se converte em maior número de chutes ou finalizações.")
