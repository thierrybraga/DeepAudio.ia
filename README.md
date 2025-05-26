DeepAudio.ia
Descrição
DeepAudio.ia é uma aplicação web baseada em Flask projetada para processar arquivos de áudio, extrair características, treinar um modelo de Rede Neural Convolucional (CNN) e detectar áudio "falso" (ou seja, áudio que não corresponde ao modelo treinado). O projeto é ideal para pesquisadores e desenvolvedores interessados em aprendizado de máquina aplicado ao processamento de áudio.
A aplicação oferece as seguintes funcionalidades:

Upload de Dataset: Permite o upload de um arquivo .zip contendo um conjunto de dados de áudio.
Extração de Características: Processa arquivos de áudio para extrair características de pitch, gerando arquivos CSV.
Treinamento do Modelo: Treina um modelo CNN com base nas características extraídas.
Detecção de Áudio Falso: Permite o upload de uma amostra de voz para verificar se ela corresponde ao modelo treinado.

Sumário

Instalação
Uso
Estrutura do Projeto
Como Funciona
Pré-requisitos
Contribuições
Licença
Contato
Agradecimentos

Instalação
Para configurar o projeto localmente, siga os passos abaixo:

Clone o repositório:
git clone https://github.com/thierrybraga/DeepAudio.ia.git


Navegue até o diretório do projeto:
cd DeepAudio.ia


Crie um ambiente virtual (recomendado):
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate


Instale as dependências listadas em requirements.txt:
pip install -r requirements.txt



Uso
Para iniciar a aplicação, execute o seguinte comando no terminal:
python app.py

Acesse a aplicação em um navegador web no endereço http://localhost:5000 (ou a porta configurada no código).
Funcionalidades da Aplicação

Upload de Dataset:

Faça o upload de um arquivo .zip contendo arquivos de áudio.
Certifique-se de que o dataset esteja organizado adequadamente para a extração de características.


Extração de Características (Pitch Extract):

Processa os arquivos de áudio para extrair características de pitch usando o script Voice2data.py.
Os resultados são salvos como arquivos CSV no diretório features/.


Treinamento do Modelo (Train Model):

Treina um modelo CNN com os dados extraídos, utilizando o script TrainModel.py.
O modelo treinado é salvo para uso na detecção.


Detecção de Áudio Falso (Fake Detector):

Faça o upload de uma amostra de voz.
A aplicação extrai características da amostra e usa o modelo treinado para classificá-la como "real" ou "falsa".



Estrutura do Projeto
O repositório está organizado da seguinte forma:



Arquivo/Diretório
Descrição



app.py
Aplicação Flask principal com rotas para upload, extração, treinamento e detecção.


Voice2data.py
Script para processar arquivos de áudio e extrair características de pitch.


TrainModel.py
Script que define e treina o modelo CNN para classificação de áudio.


Predictor.py
Script para realizar predições usando o modelo treinado.


requirements.txt
Lista de dependências do projeto.


readme.md
Documentação do projeto (este arquivo).


uploads/
Diretório para armazenar arquivos enviados (dataset e amostras de voz).


dataset/
Diretório para extrair o conteúdo do arquivo .zip do dataset.


features/
Diretório para salvar arquivos CSV com características extraídas.


templates/
Diretório com templates HTML:


  index.html
Página inicial para upload, extração e treinamento.


  fake_detector.html
Página para upload de amostras de voz e exibição de resultados.


Como Funciona

Extração de Características:

Os arquivos de áudio são processados pelo script Voice2data.py para extrair características de pitch.
As características são salvas em arquivos CSV no diretório features/.


Treinamento do Modelo:

O script TrainModel.py define a arquitetura do modelo CNN e realiza o treinamento com base nas características extraídas.
O modelo treinado é salvo para uso posterior.


Detecção de Áudio Falso:

Uma amostra de voz é processada para extrair características, da mesma forma que no treinamento.
O script Predictor.py usa o modelo treinado para classificar a amostra como "real" (corresponde ao modelo) ou "falsa" (não corresponde).



Pré-requisitos

Python 3.x: Necessário para executar a aplicação e os scripts.
Navegador Web: Para acessar a interface da aplicação Flask.
Conhecimento Básico: Familiaridade com Python e conceitos de aprendizado de máquina é recomendada para entender e modificar o código.

Contribuições
Contribuições são bem-vindas! Para contribuir:

Faça um fork do repositório.
Crie uma branch para suas alterações: git checkout -b minha-alteracao.
Envie um pull request com uma descrição clara das mudanças.

Por favor, abra um issue para discutir problemas ou sugestões antes de enviar alterações significativas.
Licença
A licença específica do projeto não foi fornecida nas informações disponíveis. Verifique o repositório para detalhes sobre a licença aplicável.
Contato
Para perguntas ou suporte, entre em contato com o mantenedor do projeto através do GitHub: thierrybraga.
Agradecimentos
Este projeto pode utilizar bibliotecas de terceiros, como:

