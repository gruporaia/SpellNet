# **Aplicação SignLink**

![](./app/example.png)

O código desenvolvido neste diretório cria uma aplicação web utilizando da biblioteca streamlit.
A simples aplicação recebe do usuário uma palavra, a qual será soletrada por meio do uso de linguagem de sinais.  
A partir da captura da imagem da webcam do usuário, são extraídos as marcações das mãos (caso estejam aparecerendo na tela), enviando para um modelo de detecção de linguagem de sinais (WIP). 
Assim que o usuário finaliza todas as letras desta palavra, o aprendizado se encerra, apresentando uma mensagem de parabenização.

### **Como executar**
Para rodar esta aplicação, **é necessário ter Docker instalado e, em caso de Docker Desktop, que ele esteja aberto.**
A partir dos seguintes comandos, todos os requisitos (dentro do código *requirements.txt* serão baixados), e você poderá visualizar a aplicação rodando localmente, na URL: http://localhost:8501

Clone o repositório
```
git clone https://github.com/gruporaia/SignLink-dev.git
```

Após clonar o repositório, navegue até a pasta **/app** (onde está o Dockerfile)
```
cd app
```

Criando uma Docker image com os requisitos do programa
```
docker build -t signlink:latest .
```

Rodando um container com a imagem criada
```
docker run -p 8501:8501 signlink:latest
```

### Próximos passos

- Investigar a combinação de diferentes técnicas de aprendizado e processamento (escala de cor, profundidade, etc.) para aprimorar a capacidade de generalização dos modelos.  
- Implementar e integrar redes LSTM para reconhecimento de gestos sequenciais, expandindo o reconhecimento além de imagens estáticas.  
- Explorar arquiteturas que unam CNNs com LSTMs para capturar tanto características espaciais quanto temporais dos sinais.  

## 💻 Quem somos
| ![LogoRAIA](https://github.com/user-attachments/assets/ce3f8386-a900-43ff-af84-adce9c17abd2) |  Este projeto foi desenvolvido pelos membros do **RAIA (Rede de Avanço de Inteligência Artificial)**, uma iniciativa estudantil do Instituto de Ciências Matemáticas e de Computação (ICMC) da USP - São Carlos. Somos estudantes que compartilham o objetivo de criar soluções inovadoras utilizando inteligência artificial para impactar positivamente a sociedade. Para saber mais, acesse [nosso site](https://gruporaia.vercel.app/) ou [nosso Instagram](instagram.com/grupo.raia)! |
|------------------|-------------------------------------------|

---

### 🤝 Parceria
| <img src="https://github.com/gruporaia/SpellNet/raw/main/images/sign_link_project_logo.jpeg" alt="SignLink" width="500"/> |  Este projeto foi desenvolvido em colaboração com a **SignLink**, uma startup brasileira encubada pela StartFellowship dedicada à promoção da acessibilidade e inclusão por meio do desenvolvimento de tecnologias para a comunicação em Línguas de Sinais. Para saber mais, acesse o [site](https://signlinkproject.github.io/index.html) ou o [LinkedIn](https://br.linkedin.com/company/sign-link-project)! |
|------------------|-------------------------------------------|

### Desenvolvedores
- **Cecilia Sedenho** - [LinkedIn](https://br.linkedin.com/in/cec%C3%ADlia-nunes-sedenho-305059255/pt) | [GitHub](https://github.com/HeNunes)
- **Joao Pedro Viguini** - [LinkedIn](https://br.linkedin.com/in/jo%C3%A3o-pedro-viguini-1829281bb) | [GitHub](https://github.com/jpviguini)
- **Daniel Carvalho** - [LinkedIn](https://br.linkedin.com/in/daniel-carvalho-aba61717a) | [GitHub](https://github.com/danielcarvalho99)
- **Bernardo Marques** - [LinkedIn](https://br.linkedin.com/in/bernardo-marques-costa) | [GitHub](https://github.com/bmarquescost)
- **Gabriel Iamato** - [LinkedIn](https://br.linkedin.com/in/gabriel-campanelli-iamato) | [GitHub](https://github.com/GabrielIamato)
- **Matheus Vicente** - [LinkedIn](https://br.linkedin.com/in/matheushrv) | [GitHub](https://github.com/MatheusHRV)

