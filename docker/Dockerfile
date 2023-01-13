FROM python:3.6

# create working directory
WORKDIR /usr/src/python_workspace

COPY . .
RUN pip install --no-cache-dir -r docker/requirements.txt

CMD ["python", "./plot.py"]
