# L-BFGS m-Variation (PyTorch)

## Descrição
Experimento simples para comparar o efeito do `history_size (m)` no L-BFGS usando um MLP suave (Softplus) no MNIST.

## Conceitos
- **Outer step:** uma chamada a `optimizer.step(closure)` → 1 atualização de pesos.  
- **Closure call:** cada avaliação de perda/gradiente dentro do line search (1 passada pelos dados).  
- **m (history_size):** quantos pares (s, y) de curvatura o L-BFGS guarda para aproximar a Hessiana.

## Como usar
```bash
python optimizer_bfgs_experiment/train.py
