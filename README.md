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
## 💻 Quem somos
| ![LogoRAIA](https://github.com/user-attachments/assets/ce3f8386-a900-43ff-af84-adce9c17abd2) |  Este projeto foi desenvolvido pelos membros do **RAIA (Rede de Avanço de Inteligência Artificial)**, uma iniciativa estudantil do Instituto de Ciências Matemáticas e de Computação (ICMC) da USP - São Carlos. Somos estudantes que compartilham o objetivo de criar soluções inovadoras utilizando inteligência artificial para impactar positivamente a sociedade. Para saber mais, acesse [nosso site](https://gruporaia.vercel.app/) ou [nosso Instagram](instagram.com/grupo.raia)! |
|------------------|-------------------------------------------|

---

### 🤝 Parceria
| ![SignLink](https://signlinkproject.github.io/images/logo.png) |  Este projeto foi desenvolvido em colaboração com a **SignLink**, uma startup brasileira encubada pela StartFellowship dedicada à promoção da acessibilidade e inclusão por meio do desenvolvimento de tecnologias para a comunicação em Línguas de Sinais. Para saber mais, acesse o [site](https://signlinkproject.github.io/index.html) ou o [LinkedIn](https://br.linkedin.com/company/sign-link-project)! |
|------------------|-------------------------------------------|

