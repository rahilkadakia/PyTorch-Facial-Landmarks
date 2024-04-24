FROM python:3.9

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

ENV PYTHONDONTWRITEBYTECODE 1
ENV FLASK_APP "app.py"
ENV FLASK_ENV "development"

EXPOSE 5000
ENTRYPOINT ["python"]
#CMD flask run --host=0.0.0.0
CMD ["app.py"]