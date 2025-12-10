import sys
from utils.visualization import plot_learning_curve

if __name__ == "__main__":
    # Passar o modelo como argumento: "ncf", "dmf" ou "ae"
    model_type = sys.argv[1] if len(sys.argv) > 1 else "ncf"
    
    plot_learning_curve(model_type)
