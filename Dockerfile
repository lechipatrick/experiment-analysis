FROM python:3.9.13 as base

WORKDIR /app

COPY poetry.lock pyproject.toml /app/

RUN pip3 install poetry==1.1.12
RUN poetry config virtualenvs.create false


FROM base
RUN poetry install
COPY . /app/
CMD ["pytest"]