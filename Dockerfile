FROM python:3.5.2

ENV JAVA_HOME=/opt/java/openjdk
COPY --from=eclipse-temurin:8-jre $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

ENV CELERY_BROKER_URL redis://redis:6379/
ENV CELERY_RESULT_BACKEND redis://redis:6379/

# copy source code
COPY . /app
WORKDIR /app

# install requirements
RUN pip3 --trusted-host pypi.python.org install -r requirements.txt

WORKDIR /app/src

# expose the app port
EXPOSE 5000

RUN python main.py generate-secret

# run the app server
CMD gunicorn --bind 0.0.0.0:5000 --workers=2 app:app
