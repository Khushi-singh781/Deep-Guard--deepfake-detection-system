FROM python:3.12-slim
WORKDIR /app
COPY backend backend
RUN pip install -r backend/requirements.txt
EXPOSE 5001
CMD ["python", "backend/app.py", "--port=5001"]
