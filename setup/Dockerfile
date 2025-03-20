FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

USER root

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y tzdata
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    perl \
    graphviz \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install TinyTeX
RUN rm -rf ~/.TinyTeX && \
    wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh && \
    echo 'export PATH="$PATH:$HOME/.TinyTeX/bin/x86_64-linux"' >> ~/.bashrc && \
    . ~/.bashrc

# Install TeX packages
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-pip

# Copy requirements file
COPY requirements_gpu.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_gpu.txt --no-deps

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# # Create a non-root user
# RUN useradd -m -s /bin/bash developer
# RUN chown -R developer:developer /app

# # Switch to non-root user
# USER developer

# Set up bash as default shell with useful aliases
RUN echo 'alias ll="ls -la"' >> ~/.bashrc && \
    echo 'alias python="python3"' >> ~/.bashrc

# Start interactive bash shell by default
CMD ["/bin/bash"]
