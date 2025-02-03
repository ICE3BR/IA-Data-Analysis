import io
import sys

import pandas as pd
import streamlit as st
from langchain_ollama import (
    OllamaLLM,  # Importa a classe OllamaLLM do pacote langchain-ollama
)
from pandasai import SmartDataframe

# Configurações iniciais da interface Streamlit
st.title("🚀 Data Analysis with PandasAI")


# Função para carregar e ajustar os dados do CSV
@st.cache_data
def load_data(file):
    """
    Carrega e ajusta os dados do arquivo CSV.

    Args:
        file (UploadedFile): Arquivo CSV carregado pelo usuário.

    Returns:
        pd.DataFrame: DataFrame com os dados carregados e colunas ajustadas.
    """
    data = pd.read_csv(file, encoding="utf-8")
    data.columns = data.columns.astype(
        str
    ).str.strip()  # Remove espaços extras nos nomes das colunas
    return data


# Função para exibir os primeiros registros do DataFrame
def display_data(data):
    """
    Exibe os primeiros registros do DataFrame.

    Args:
        data (pd.DataFrame): DataFrame com os dados carregados.
    """
    st.write("📊 **Visualizando os primeiros dados:**")
    st.write(data.head(3))  # Mostra as primeiras 3 linhas do DataFrame


# Função para processar o prompt do usuário e retornar a resposta
def process_prompt(df, prompt):
    """
    Processa o prompt do usuário usando o SmartDataframe.

    Args:
        df (SmartDataframe): DataFrame inteligente configurado com o modelo LLM.
        prompt (str): Comando ou pergunta do usuário.

    Returns:
        tuple: Resposta gerada pelo modelo e texto capturado do terminal.
    """
    with st.spinner("⏳ Gerando resposta..."):
        try:
            # Captura a saída padrão para redirecionar para uma variável
            captured_output = io.StringIO()
            sys.stdout = captured_output

            # Chama o método chat do SmartDataframe
            response = df.chat(prompt)

            # Restaura a saída padrão
            sys.stdout = sys.__stdout__

            # Obtém o texto capturado
            output_text = captured_output.getvalue().strip()
            return response, output_text
        except Exception as e:
            # Restaura a saída padrão caso ocorra erro
            sys.stdout = sys.__stdout__
            st.error(f"❌ Erro ao processar o prompt: {e}")
            st.write("🔍 Verifique os nomes das colunas:", df.dataframe.columns)
            return None, None


# Upload de arquivo CSV
uploader_file = st.file_uploader("⬆️ Upload um arquivo CSV", type=["csv"])

# Verifica se um arquivo foi carregado
if uploader_file is not None:
    try:
        # Carrega e exibe os dados
        data = load_data(uploader_file)
        display_data(data)

        # Configura o modelo LLM e cria o SmartDataframe
        llm = OllamaLLM(model="deepseek-r1:8b")  # Use a classe OllamaLLM
        df = SmartDataframe(data, config={"llm": llm, "cache": None})
    except Exception as e:
        # Exibe mensagem de erro caso falhe ao carregar o arquivo
        st.error(f"❌ Erro ao carregar o arquivo: {e}")
        st.stop()

# Entrada de prompt do usuário com exemplo
prompt = st.text_area(
    "📝 Digite seu comando:",
    value="Com apenas os dados existentes. Liste os 5 países com a menor população em ordem crescente. Retorne uma tabela formatada.",
)

# Botão para gerar resposta
if st.button("🚀 Gerar Resposta"):
    # Verifica se o prompt não está vazio
    if prompt.strip():
        # Processa o prompt e obtém a resposta e o texto capturado
        response, output_text = process_prompt(df, prompt)

        # Placeholder para exibir a resposta
        response_placeholder = st.empty()

        # Verifica se há uma resposta
        if response is not None:
            # Se a resposta for um DataFrame, exibe a tabela formatada
            if isinstance(response, pd.DataFrame):
                response_placeholder.write("📊 **Tabela Resultante:**")
                response_placeholder.dataframe(response)
            # Se a resposta for um caminho de imagem, exibe o gráfico
            elif isinstance(response, str) and response.endswith(".png"):
                response_placeholder.image(
                    response,
                    caption="📊 Gráfico gerado pelo PandasAI",
                    use_container_width=True,
                )
            else:
                # Caso contrário, exibe a resposta como texto
                response_placeholder.write(response)

        # Exibe o texto capturado do terminal
        if output_text:
            st.text(output_text)
    else:
        # Exibe aviso se o prompt estiver vazio
        st.warning("⚠️ O comando não pode estar vazio!")
