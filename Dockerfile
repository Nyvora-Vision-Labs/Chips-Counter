FROM python:3.11-slim

# Install system dependencies needed for OpenCV and rembg
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "rembg[cpu]"

# Copy the rest of the application files
COPY --chown=user . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port 5050
EXPOSE 5050

# Run the Flask server
CMD ["python", "server.py"]
