# Use the official Python 3.10 image as a base
# FROM python:3.10-slim
FROM mcr.microsoft.com/devcontainers/python:1-3.10-bullseye


# Install necessary tools for C/C++ development
RUN apt-get update && apt-get install -y \
    build-essential \
    gdb \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Optionally, you can install additional tools or libraries here

# Set the working directory in the container
WORKDIR /workspace

# Set this as the default directory when the container starts
CMD ["bash"]
