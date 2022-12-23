FROM pytorch/pytorch:1.13.0-cpu
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt