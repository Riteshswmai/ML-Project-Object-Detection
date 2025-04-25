# Step 1: Use an official Python runtime as the base image
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install any system dependencies if needed (for example, OpenCV dependencies)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy the requirements file into the container
COPY requirements.txt /app/

# Step 5: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your project files into the container
COPY . /app/

# Step 7: Set the entry point to run your object detection tracking script
CMD ["python", "object.py"]
