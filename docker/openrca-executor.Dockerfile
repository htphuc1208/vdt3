FROM python:3.12-slim

RUN pip install --no-cache-dir "pandas>=2.0" "numpy>=1.26" "pyarrow>=14.0" pytz
RUN useradd --create-home --uid 10001 sandbox
WORKDIR /sandbox
COPY telco_mas/openrca/sandbox_kernel.py /sandbox/sandbox_kernel.py
USER sandbox
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/sandbox/sandbox_kernel.py"]
