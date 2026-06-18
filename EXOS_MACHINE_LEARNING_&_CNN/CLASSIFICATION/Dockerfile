FROM python:3.12.7

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", "mlp_bank_customer_churn_prediction.py" ]