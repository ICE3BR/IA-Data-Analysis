import io
import sys

import pandas as pd
import streamlit as st
from langchain_community.llms import Ollama
from pandasai import SmartDataframe

# Configurar o modelo LLM
llm = Ollama(model="mistral:latest")

st.title("Data Analysis with PandasAI")

# Upload de arquivo CSV
uploader_file = st.file_uploader("Upload um arquivo CSV", type=["csv"])

if uploader_file is not None:
    # Carregar o CSV
    data = pd.read_csv(uploader_file, encoding="utf-8")

    # Ajustar colunas (remover espaços extras)
    data.columns = data.columns.astype(str).str.strip()

    st.write("📊 **Visualizando os primeiros dados:**")
    st.write(data.head(3))  # Exibir os primeiros registros

    # Criar SmartDataframe
    df = SmartDataframe(data, config={"llm": llm, "cache": None})

    # Entrada de prompt do usuário
    prompt = st.text_area("Digite seu comando:")

    if st.button("Gerar Resposta"):
        if prompt:
            with st.spinner("Gerando resposta..."):
                try:
                    # Capturar a saída que normalmente iria para o terminal
                    captured_output = io.StringIO()
                    sys.stdout = captured_output  # Redireciona a saída para a variável

                    response = df.chat(prompt)  # Chama o PandasAI

                    sys.stdout = sys.__stdout__  # Restaura a saída normal do terminal

                    # Obter o texto capturado
                    output_text = captured_output.getvalue().strip()

                    st.write("📝 **Resposta:**")

                    # Se a resposta for um caminho de imagem, exibe o gráfico
                    if isinstance(response, str) and response.endswith(".png"):
                        st.image(
                            response,
                            caption="📊 Gráfico gerado pelo PandasAI",
                            use_container_width=True,
                        )

                    # Exibir texto capturado do terminal no Streamlit
                    if output_text:
                        st.text(output_text)

                    # Se `df.chat()` retornar uma resposta de texto, exibir no Streamlit
                    if isinstance(response, str) and not response.endswith(".png"):
                        st.write(response)

                except Exception as e:
                    sys.stdout = (
                        sys.__stdout__
                    )  # Restaura a saída normal do terminal caso ocorra erro
                    st.error(f"Erro ao processar o prompt: {e}")
                    st.write("Verifique os nomes das colunas:", data.columns)
        else:
            st.warning("Por favor, digite um comando!")
