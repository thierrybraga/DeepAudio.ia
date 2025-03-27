# Projeto de Análise de Áudio com Flask

## Descrição

Este projeto tem como objetivo realizar a extração de features de áudio, treinamento de um modelo de rede neural convolucional (CNN) e detecção de “fake” (áudio que não corresponde ao modelo treinado). A aplicação web, desenvolvida com Flask, oferece as seguintes funcionalidades:

- **Upload de Dataset:** Permite o upload de um arquivo `.zip` contendo um dataset de áudios.
- **Pitch Extract:** Processa os arquivos de áudio do dataset utilizando o script `Voice2data.py` para extrair features e gerar arquivos CSV.
- **Train Model:** Treina um modelo CNN usando o script `TrainModel.py` com os dados extraídos. O modelo treinado será utilizado pelo `Predictor.py`.
- **Fake Detector:** Permite o upload de uma amostra de voz, extrai suas features e realiza a predição para identificar se a voz corresponde ou não ao modelo treinado.

## Estrutura do Projeto

meu_projeto_flask/ ├── app.py # Aplicação Flask principal, com as rotas para upload, extração, treinamento e detecção ├── Voice2data.py # Processa arquivos de áudio e extrai features (Pitch Extract) ├── TrainModel.py # Define e treina o modelo CNN para classificação de áudio ├── Predictor.py # Realiza a predição utilizando o modelo treinado ├── requirements.txt # Lista de dependências do projeto ├── readme.md # Documentação detalhada do projeto ├── uploads/ # Diretório para armazenamento de arquivos enviados (dataset e amostras de voz) ├── dataset/ # Diretório para extração do conteúdo do arquivo .zip do dataset ├── features/ # Diretório para salvamento dos arquivos CSV com features extraídas └── templates/ # Diretório com os templates HTML da aplicação ├── index.html # Página inicial com as funções de upload, extração e treinamento └── fake_detector.html # Página para upload de amostra de voz e exibição do resultado da predição


## Pré-requisitos

- Python 3.x
- (Opcional) Ambiente virtual para isolar as dependências

## Instalação

1. **Clone o repositório ou faça o download dos arquivos do projeto:**
   ```bash
   git clone <URL_do_repositorio>
   cd meu_projeto_flask
   python -m venv venv


