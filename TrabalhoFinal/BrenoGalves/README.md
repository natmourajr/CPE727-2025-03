# Stock Market Prediction

This project uses Deep Learning models (CNN, LSTM, Transformer) to predict stock prices.

## Project Structure

- `src/stock_market_prediction.ipynb`: The main Jupyter Notebook containing the training and evaluation logic.
- `src/apresentacao_Breno_Galves.pdf`: A presentation of the project.
- `docker/Dockerfile`: Defines the Docker environment with Python 3.9 and necessary libraries.
- `requirements.txt`: List of Python dependencies.

## How to Run

1.  **Navigate to the `docker` directory:**
    ```bash
    cd docker
    ```

2.  **Build and Start the Container:**
    ```bash
    docker-compose up --build
    ```

2.  **Access JupyterLab:**
    Open your browser and navigate to:
    [http://localhost:8888](http://localhost:8888)

3.  **Run the Prediction:**
    Open `stock_market_prediction.ipynb` inside JupyterLab and run all cells.

## Dependencies

- torch
- numpy
- pandas
- yfinance
- scikit-learn
- tabulate

## Troubleshooting

### `ModuleNotFoundError: No module named 'distutils'`

If you encounter this error when running `docker-compose`, it is likely because you are using Python 3.12, which has removed `distutils`. The version of `docker-compose` you are using is not compatible with Python 3.12.

To fix this, you can either:

1.  **Install `setuptools`**: This will provide the missing `distutils` module.
    ```bash
    pip install setuptools
    ```
    If you are using a system-managed Python installation, you might need to use your system's package manager, for example:
    ```bash
    sudo apt-get install python3-setuptools
    ```

2.  **Upgrade `docker-compose`**: It is recommended to use the latest version of `docker-compose` (v2), which is included with Docker Desktop or can be installed as a Docker plugin.
