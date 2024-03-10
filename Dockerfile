FROM ubuntu:latest

# Install Python and LibreOffice
RUN apt-get update && \
    apt-get install -y python3 python3-pip libreoffice poppler-utils && \
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

# Install Node.js
RUN apt-get update && \
    apt-get install -y curl && \
    curl -sL https://deb.nodesource.com/setup_21.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Marp CLI
RUN npm install --save-dev @marp-team/marp-cli
# Set the entry point
ENTRYPOINT ["python3"]
