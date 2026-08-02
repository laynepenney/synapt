FROM python:3.12-slim

WORKDIR /opt/synapt

COPY pyproject.toml README.md ./
COPY src ./src
COPY demo ./demo

# The spike exercises recall's existing archive and search code without the
# heavyweight optional model stack. pyrage is the real age implementation.
RUN pip install --no-cache-dir --no-deps -e . \
    && pip install --no-cache-dir pyrage==1.3.0

ENV PYTHONPATH=/opt/synapt:/opt/synapt/src \
    PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
