ARG RAY_TAG_VERSION="2.7.1-py310"
ARG FULL_BASE_IMAGE=rayproject/ray:"$RAY_TAG_VERSION"

FROM $FULL_BASE_IMAGE

ENV VERSION 0.1.0

COPY requirements.txt .

RUN pip install -U requirements.txt

COPY src/ .