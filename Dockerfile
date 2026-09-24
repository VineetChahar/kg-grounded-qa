FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY tests ./tests
RUN pip install --no-cache-dir -e .

COPY scripts ./scripts

CMD ["python", "-m", "kg_grounded_qa.run_eval"]
