# Use an official PyTorch image as the base image
FROM pytorch/pytorch:latest

# install git
RUN apt-get update && apt-get install -y git

RUN pip install torch==2.6.1

# Install any additional packages you need
RUN pip install ray[tune] scikit-learn pytorch-lightning

# Set the working directory
WORKDIR /workspace

# Start a shell by default
CMD ["bash"]
