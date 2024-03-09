FROM ubuntu:latest

# Install Python and LibreOffice
RUN apt-get update && \
    apt-get install -y python3 python3-pip libreoffice libreoffice-writer libreoffice-calc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies including pptx module
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python scripts into the container
COPY . /app

# Set the entry point
ENTRYPOINT ["python3"]
