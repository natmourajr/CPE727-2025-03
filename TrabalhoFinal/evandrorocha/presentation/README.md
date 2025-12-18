# Apresentação - Detecção de Tuberculose

Apresentação em LaTeX (Beamer) sobre o projeto de detecção de tuberculose usando Deep Learning.

## 📋 Estrutura

A apresentação está organizada em 6 seções principais:

1. **Introdução** - Contexto e motivação
2. **Problema Abordado** - Definição do problema e dataset
3. **Revisão Bibliográfica** - Estado da arte e base teórica
4. **Método Proposto** - Pipeline e modelos implementados
5. **Resultados Obtidos** - Métricas e análises
6. **Conclusões** - Contribuições e trabalhos futuros

## 🔧 Como Compilar

### Opção 1: Online (Overleaf)

1. Acesse [Overleaf](https://www.overleaf.com/)
2. Crie um novo projeto
3. Faça upload do arquivo `apresentacao.tex`
4. Compile (Ctrl+S ou botão Recompile)

### Opção 2: Local (LaTeX instalado)

```bash
# Compilar com pdflatex
pdflatex apresentacao.tex
pdflatex apresentacao.tex  # Segunda vez para referências

# OU com latexmk (recomendado)
latexmk -pdf apresentacao.tex
```

### Opção 3: Docker

```bash
# Usando imagem LaTeX
docker run --rm -v ${PWD}:/workspace -w /workspace \
    texlive/texlive:latest \
    pdflatex apresentacao.tex
```

## ✏️ Personalização

### Informações Pessoais

Edite as linhas 14-17:

```latex
\title{Detecção de Tuberculose em Radiografias de Tórax}
\subtitle{Utilizando Deep Learning e Redes Neurais Convolucionais}
\author{SEU NOME AQUI}
\institute{SUA UNIVERSIDADE}
```

### Adicionar Resultados

Quando o treinamento terminar, preencha a tabela na seção "Resultados Obtidos" (slide ~linha 280):

```latex
ResNet-50 & 0.XXX & 0.XXX & 0.XXX & 0.XXX & 0.XXX \\
```

Substitua `0.XXX` pelos valores reais de:
- Acurácia
- Precisão
- Recall
- F1-Score
- AUC-ROC

### Adicionar Gráficos

Para incluir gráficos (curvas ROC, matriz de confusão, etc.):

```latex
\begin{figure}
    \centering
    \includegraphics[width=0.8\textwidth]{caminho/para/grafico.png}
    \caption{Descrição do gráfico}
\end{figure}
```

Coloque as imagens na mesma pasta que o `.tex` ou em uma subpasta `figures/`.

## 🎨 Temas Alternativos

Para mudar o tema visual, edite a linha 2:

```latex
% Temas disponíveis:
\usetheme{Madrid}      % Atual
\usetheme{Berlin}      % Moderno
\usetheme{Copenhagen}  % Minimalista
\usetheme{Warsaw}      % Clássico
```

Cores:

```latex
\usecolortheme{default}  % Atual
\usecolortheme{beaver}   % Vermelho
\usecolortheme{dolphin}  # Azul
\usecolortheme{orchid}   # Roxo
```

## 📊 Slides Importantes

- **Slide 1-2:** Título e sumário
- **Slide 3-4:** Introdução e motivação
- **Slide 5-6:** Definição do problema
- **Slide 7-8:** Revisão bibliográfica
- **Slide 9-11:** Método proposto
- **Slide 12-14:** Resultados (PREENCHER!)
- **Slide 15-17:** Conclusões e trabalhos futuros

## ⏱️ Timing (15 minutos)

Sugestão de distribuição de tempo:

- Introdução: 2 min
- Problema: 2 min
- Revisão: 3 min
- Método: 4 min
- Resultados: 3 min
- Conclusões: 1 min

## 📝 Checklist Antes da Apresentação

- [ ] Preencher nome e instituição
- [ ] Adicionar resultados reais do treinamento
- [ ] Incluir gráficos (ROC, confusion matrix)
- [ ] Revisar todas as referências
- [ ] Testar compilação
- [ ] Praticar apresentação (15 min)
- [ ] Preparar respostas para perguntas comuns

## 🔗 Links Úteis

- [Beamer User Guide](https://ctan.org/pkg/beamer)
- [Overleaf Beamer Templates](https://www.overleaf.com/gallery/tagged/presentation)
- [LaTeX Color Names](https://www.overleaf.com/learn/latex/Using_colours_in_LaTeX)

## 💡 Dicas

1. **Mantenha slides simples** - Não sobrecarregue com texto
2. **Use imagens** - Gráficos são mais impactantes que tabelas
3. **Pratique** - Ensaie a apresentação várias vezes
4. **Tempo** - Deixe 2-3 minutos para perguntas
5. **Backup** - Tenha PDF pronto em pen drive e nuvem
