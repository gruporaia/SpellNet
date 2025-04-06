# **Aplicação SignLink**

![](https://raw.githubusercontent.com/gruporaia/SignLink-dev/refs/heads/main/app/example.png?token=GHSAT0AAAAAADBK6Q3U5KFQZKNWFHQULLCWZ7S2P2Q)

O código desenvolvido neste diretório cria uma aplicação web utilizando da biblioteca streamlit.
A simples aplicação recebe do usuário uma palavra, a qual será soletrada por meio do uso de linguagem de sinais.  
A partir da captura da imagem da webcam do usuário, são extraídos as marcações das mãos (caso estejam aparecerendo na tela), enviando para um modelo de detecção de linguagem de sinais (WIP). 
Assim que o usuário finaliza todas as letras desta palavra, o aprendizado se encerra, apresentando uma mensagem de parabenização.

### **Como rodar**
Para rodar esta aplicação, o único requisito é ter Docker instalado.
A partir dos seguintes comandos, todos os requisitos (dentro do código *requirements.txt* serão baixados), e você poderá visualizar a aplicação rodando localmente, na URL: http://localhost:8501

Criando uma Docker image com os requisitos do programa
```
docker build -t signlink:latest .
```

Rodando um container com a imagem criada
```
docker run -p 8501:8501 signlink:latest
```
