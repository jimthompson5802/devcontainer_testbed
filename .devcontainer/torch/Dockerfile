# Use an official PyTorch image as the base image
FROM pytorch/pytorch:latest

# install git
RUN apt-get update && apt-get install -y git net-tools

# Install any additional packages you need
RUN pip install ray[air] scikit-learn pytorch-lightning torchinfo \
    ipywidgets matplotlib seaborn pandas ipykernel \
    tensorboard 

# Set the working directory
WORKDIR /workspace

# Start a shell by default
CMD ["bash"]
