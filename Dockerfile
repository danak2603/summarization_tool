# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make sure output appears immediately (useful for logs)
ENV PYTHONUNBUFFERED=1

# Run the app (adjust if main.py is in a subfolder)
CMD ["python", "main.py"]
