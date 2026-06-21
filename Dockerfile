# Usa uma imagem oficial leve do Python
FROM python:3.12-slim

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia os arquivos de dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o resto do projeto (incluindo a pasta src e o modelo_rf.pkl)
COPY . .

# Expõe a porta padrão que o Streamlit usa na nuvem
EXPOSE 8080

# Comando para rodar o Streamlit configurando a porta para o Cloud Run
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]