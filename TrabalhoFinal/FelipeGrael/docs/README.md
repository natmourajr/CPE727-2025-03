# Autoencoders

Este repositório contém os arquivos para a apresentação sobre Autoencoders da disciplina CPE727, apresentada em 2025/3.

## Como compilar

Este repositório provê três métodos para montagem do ambiente necessário para compilar: usando Devbox, Dev Containers ou construindo diretamente com o LaTeX da sua própria máquina.

### Usando Devbox

[Devbox](https://www.jetify.com/devbox) é uma ferramenta _open source_ para criação de ambientes de desenvolvimento isolados e reprodutíveis, baseado em Nix. Para usá-lo basta seguir os seguintes passos:

1. Instalar o Devbox, de acordo com as [instruções de instalação](https://www.jetify.com/docs/devbox/installing-devbox)
2. Executar o comando para compilar:

```shell
devbox run task all
```

Na primeira vez que esse comando for executado, o Devbox irá baixar e configurar o ambiente necessário, o que pode levar alguns minutos. Após isso, o comando irá compilar a apresentação. O PDF resultante estará disponivel em `autoencoders.pdf`

Também é possível entrar em um shell interativo dentro do ambiente do Devbox, executando:

```shell
devbox shell
```

Nesse shell, você pode executar o comando `task all` para compilar a apresentação. Também terá acesso a todas as ferramentas e dependências instaladas no ambiente do Devbox.

### Usando Dev Containers

Caso esteja utilizando o Visual Studio Code, ou outro editor compatível, você pode usar a funcionalidade de Dev Containers para criar um ambiente de desenvolvimento isolado em um container Docker. Para isso, siga os seguintes passos:

1. Certifique-se de ter o [Docker](https://www.docker.com/get-started) e a extensão [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) instalados
2. Abra o repositório no Visual Studio Code
3. Na barra inferior direita, clique no ícone verde do Dev Containers e selecione "Reopen in Container"

Na primeira vez que o container for criado, o processo pode ser bem demorado, levando muitos minutos, pois o ambiente necessário será baixado e configurado. Após isso, você poderá compilar a apresentação executando o seguinte comando no terminal do container:

```shell
task all
```

Quando o Dev Container estiver ativo, a janela de shell do VSCode estará dentro do ambiente do container, e você terá acesso a todas as ferramentas e dependências instaladas nele.

### Usando localmente

Caso você já possua uma instalação do LaTeX completa na sua máquina, você pode compilar a apresentação diretamente. Para isso, basta executar o seguinte comando:

```shell
latexmk -pdf -bibtex beamerpolito.tex
```

Sugere-se, no entanto, o uso de Devbox ou Dev Containers para garantir que todas as dependências estejam instaladas e configuradas de uma forma padrão.
