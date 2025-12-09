FROM python:3.12-slim

WORKDIR /models

ENV PIP_NO_VERIFY_CERTS=1

ENV PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org download.pytorch.org"

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host download.pytorch.org \
    -r requirements.txt

COPY . .

CMD ["python", "run_ae_with_dnn.py"]
