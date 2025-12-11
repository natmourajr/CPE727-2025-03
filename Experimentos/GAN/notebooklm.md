# Resumo do artigo principal

As Redes Adversariais Generativas, ou GANs (Generative Adversarial Networks), representam uma nova estrutura para a estimação de modelos generativos por meio de um processo adversarial. O funcionamento das GANs baseia-se em um cenário de teoria dos jogos de soma zero, no qual dois modelos competem simultaneamente.

## Funcionamento das GANs

O funcionamento das GANs envolve o treinamento simultâneo de dois modelos, geralmente definidos por multilayer perceptrons (MLPs), conhecidos como redes adversariais:

1. Modelo Generativo ($G$): Este modelo tem como objetivo capturar a distribuição dos dados ($p_{\text{data}}$). Ele gera amostras passando um **ruído de entrada aleatório** ($z \sim p_z(z)$) através de uma função diferenciável $G(z; \theta_g)$.
2. Modelo Discriminativo ($D$): Este modelo atua como um adversário e estima a probabilidade de que uma amostra tenha vindo dos dados de treinamento em vez de ter sido gerada por $G$. $D(x; \theta_d)$ é um escalar que representa a probabilidade de $x$ ser um exemplo de treinamento real.

## A Analogia da Competição

A relação entre $G$ e $D$ é análoga a um jogo de competição:

*   O modelo generativo ($G$) é semelhante a uma **equipe de falsificadores** que tenta produzir dinheiro falso e usá-lo sem ser detectado.
*   O modelo discriminativo ($D$) é semelhante à **polícia** que tenta detectar a moeda falsificada.

A competição neste jogo força ambas as equipes a aprimorar seus métodos até que as falsificações se tornem **indistinguíveis dos artigos genuínos**.

### O Objetivo do Treinamento

O procedimento de treinamento define os objetivos de cada modelo:

*   **Treinamento de $D$:** $D$ é treinado para **maximizar a probabilidade** de atribuir o rótulo correto, classificando corretamente tanto os exemplos de treinamento (reais) quanto as amostras geradas por $G$ (falsas).
*   **Treinamento de $G$:** $G$ é treinado simultaneamente para **maximizar a probabilidade de $D$ cometer um erro**. Formalmente, $G$ é treinado para **minimizar $\log(1 - D(G(z)))$**.

### Convergência

Se $G$ e $D$ tiverem capacidade suficiente, eles atingirão um ponto de equilíbrio onde $p_g = p_{\text{data}}$ (a distribuição gerada é igual à distribuição dos dados). Neste ponto, o discriminador não consegue diferenciar entre as duas distribuições, e sua saída será **$D(x) = 1/2$** em todos os lugares.

O sistema inteiro pode ser treinado usando apenas o algoritmo de **backpropagation**. O treinamento é iterativo, alternando entre $k$ passos de otimização de $D$ e um passo de otimização de $G$.

---

## 2. A Matemática por Trás das GANs (O Jogo Minimax)

O funcionamento das GANs é formalizado como um **jogo minimax de dois jogadores**. A função valor, $V(G, D)$, governa o *payoff* do discriminador.

### A Função Valor Minimax

O jogo que $D$ e $G$ jogam é definido pela seguinte função valor $V(G, D)$, onde $G$ tenta minimizar o valor e $D$ tenta maximizar o valor:

$$\min_G \max_D V(D,G) = E_{x \sim p_{\text{data}}(x)}[\log D(x)] + E_{z \sim p_z(z)}[\log(1-D(G(z)))] \quad \text{(Equação 1)}$$

O primeiro termo, $E_{x \sim p_{\text{data}}(x)}[\log D(x)]$, representa a expectativa de que o discriminador classifique corretamente os dados reais. O segundo termo, $E_{z \sim p_z(z)}[\log(1-D(G(z)))]$, representa a expectativa de que o discriminador classifique corretamente as amostras falsas (minimizar este termo para $D$ significa maximizar a confiança de que as amostras são falsas).

### O Discriminador Ótimo ($D^*_G$)

Para qualquer gerador $G$ fixo, é possível encontrar o discriminador ótimo $D^*_G$. O critério de treinamento para $D$ é maximizar $V(G, D)$.

O valor da função $V(G, D)$ pode ser reescrito como uma integral:
$$V(G,D) = \int_x p_{\text{data}}(x) \log(D(x)) + p_g(x) \log(1-D(x))dx \quad \text{(Equação 3)}$$

O discriminador ótimo $D^*_G(x)$ é dado por (Proposição 1):

$$D^*_G(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \quad \text{(Equação 2)}$$

A maximização do objetivo de $D$ pode ser interpretada como a maximização da log-verossimilhança para estimar a probabilidade condicional $P(Y=y|x)$, onde $Y$ indica se $x$ veio de $p_{\text{data}}$ ($y=1$) ou $p_g$ ($y=0$).

### Otimização Global e Divergência Jensen-Shannon

Ao substituir o $D^*_G(x)$ ótimo de volta na função valor, obtém-se o critério de treinamento virtual $C(G)$:
$$C(G) = \max_D V(G,D)$$

O Teorema 1 demonstra que o **mínimo global** de $C(G)$ é alcançado **se e somente se $p_g = p_{\text{data}}$**. Nesse ponto, $C(G)$ atinge o valor de $-\log 4$.

A função $C(G)$ pode ser relacionada à **Divergência Jensen–Shannon (JSD)** entre a distribuição do modelo ($p_g$) e a distribuição de geração de dados ($p_{\text{data}}$):

$$C(G) = -\log(4) + 2 \cdot JSD (p_{\text{data}} \| p_g) \quad \text{(Equação 6)}$$

Como a JSD é sempre não negativa e é zero apenas quando as duas distribuições são iguais ($p_g = p_{\text{data}}$), isso prova que $C^* = -\log(4)$ é o mínimo global de $C(G)$, e a única solução é quando o modelo generativo replica perfeitamente o processo de geração de dados.

### Otimização

Na prática, a otimização usa o gradiente estocástico em minibatches (Algoritmo 1).

1.  **Atualização de $D$**: $D$ é atualizado subindo o gradiente estocástico:
    $$\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m \left[ \log D(x^{(i)}) + \log (1-D(G(z^{(i)}))) \right]$$
2.  **Atualização de $G$**: $G$ é atualizado descendo seu gradiente estocástico (tentando minimizar $\log (1-D(G(z)))$).

Um detalhe importante é que a equação 1 nem sempre fornece gradiente suficiente para $G$ aprender bem no início do treinamento, pois $\log(1 - D(G(z)))$ satura quando $D$ rejeita as amostras de $G$ com alta confiança. Uma abordagem prática é treinar $G$ para maximizar $\log D(G(z))$ (em vez de minimizar $\log(1 - D(G(z)))$), pois essa função objetivo fornece gradientes muito mais fortes no início do aprendizado, mantendo o mesmo ponto fixo das dinâmicas de $G$ e $D$.

### Analogia

O processo das GANs é como se fosse um desafio de falsificação de documentos. O Gerador ($G$) tenta criar a cópia perfeita de um documento ($p_{\text{data}}$), e o Discriminador ($D$) é o juiz. O $G$ recebe feedback apenas sobre quão bom foi o seu disfarce (se o juiz se enganou), e não sobre a receita original do documento. O jogo só termina quando o juiz ($D$) só consegue adivinhar a origem do documento 50% das vezes, indicando que a falsificação é indistinguível do original.