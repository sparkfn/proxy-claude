FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proxy.py config.py cli.py container.py ./
COPY providers/ ./providers/
COPY proxy_v2/ ./proxy_v2/

EXPOSE 2555

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import http.client; c=http.client.HTTPConnection('localhost',2555,timeout=2); c.request('GET','/health'); r=c.getresponse(); exit(0 if r.status == 200 else 1)"

CMD ["python", "proxy.py", "2555"]
