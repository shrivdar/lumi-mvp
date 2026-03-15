FROM python:3.11-slim
WORKDIR /tool
RUN groupadd -r mcp && useradd -r -g mcp -d /tool -s /sbin/nologin mcp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ../../shared /tool/shared
COPY . .
ENV PYTHONPATH=/tool/shared:/tool
RUN chown -R mcp:mcp /tool
USER mcp
EXPOSE 8080
CMD ["python", "server.py"]
