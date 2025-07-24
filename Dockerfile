# ────────────────────────────────────────────────────────────────
# CUDA 12.4 tool‑chain ‑ Ubuntu 22.04
# ────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Build‑time variable propagated from docker‑compose
ARG PROJECT=irail:ct_fm
ENV PROJECT=${PROJECT}

ARG UID=1000
ARG GID=1000
ARG USERNAME=user
ARG PYTHON_VERSION=3.10            # 22.04 ships 3.10 natively
ENV DEBIAN_FRONTEND=noninteractive
ARG PROJECT_ROOT=/opt/project

# ────────────────────────────────────────────────────────────────
# System + Python
# ────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
        python3-venv python3-pip \
        build-essential git git-lfs curl ca-certificates wget gpg \
        ninja-build cmake \
        libopenmpi-dev openmpi-bin \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
        tzdata openssh-client sudo tmux && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

# ────────────────────────────────────────────────────────────────
# Intel oneAPI CCL for DeepSpeed
# ────────────────────────────────────────────────────────────────
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
    tee /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-oneapi-ccl-devel && \
    rm -rf /var/lib/apt/lists/*

# Set up oneAPI environment
ENV CPATH=/opt/intel/oneapi/ccl/latest/include:$CPATH \
    LIBRARY_PATH=/opt/intel/oneapi/ccl/latest/lib:$LIBRARY_PATH \
    LD_LIBRARY_PATH=/opt/intel/oneapi/ccl/latest/lib:$LD_LIBRARY_PATH

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ────────────────────────────────────────────────────────────────
# Python dependencies from requirements.txt
# ────────────────────────────────────────────────────────────────
ENV PIP_DEFAULT_TIMEOUT=120          \
    PIP_RETRIES=10                   \
    PIP_NO_INPUT=1                   \
    TORCH_CUDA_ARCH_LIST=8.6         \
    DS_BUILD_OPS=1                   \
    FLASH_ATTENTION_FORCE_CUDA=1

# BuildKit cache keeps partly‑downloaded wheels
# NB: requires Docker 20.10+ with BuildKit (enabled by default on Docker 24)
RUN --mount=type=cache,sharing=locked,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN --mount=type=cache,sharing=locked,target=/root/.cache/pip \
    python3 -m pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────────────────────
# CUDA build flags
# ────────────────────────────────────────────────────────────────
ENV TORCH_CUDA_ARCH_LIST=8.6 \
    DS_BUILD_OPS=1 \
    FLASH_ATTENTION_FORCE_CUDA=1

# ────────────────────────────────────────────────────────────────
# Non‑root dev user (matches your host UID/GID)
# ────────────────────────────────────────────────────────────────
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd  -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} && \
    usermod  -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    # Verify sudo access works
    su - ${USERNAME} -c "sudo -n true" && \
    echo "✓ Sudo privileges configured successfully for ${USERNAME}"

RUN mkdir -p ${PROJECT_ROOT} \
    && chown -R ${USERNAME}:${USERNAME} ${PROJECT_ROOT} \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

# ────────────────────────────────────────────────────────────────
# After creating the non‑root user
# ────────────────────────────────────────────────────────────────
RUN mkdir -p /home/${USERNAME}/.cursor-server \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cursor-server

# Create a startup script to fix permissions
RUN echo '#!/bin/bash' > /usr/local/bin/fix-permissions.sh && \
    echo 'if [ -d "${PROJECT_ROOT}" ]; then' >> /usr/local/bin/fix-permissions.sh && \
    echo '    sudo chown -R ${USERNAME}:${USERNAME} ${PROJECT_ROOT}' >> /usr/local/bin/fix-permissions.sh && \
    echo '    sudo chmod -R u+rw ${PROJECT_ROOT}' >> /usr/local/bin/fix-permissions.sh && \
    echo '    find ${PROJECT_ROOT} -type d -exec sudo chmod u+rwx {} \;' >> /usr/local/bin/fix-permissions.sh && \
    echo 'fi' >> /usr/local/bin/fix-permissions.sh && \
    echo 'exec "$@"' >> /usr/local/bin/fix-permissions.sh && \
    chmod +x /usr/local/bin/fix-permissions.sh

USER ${USERNAME}
WORKDIR ${PROJECT_ROOT}

# Use the permission fix script as entrypoint
ENTRYPOINT ["/usr/local/bin/fix-permissions.sh"]

# VS Code / Cursor will upload its own server here; no need to pre‑create
CMD [ "/bin/bash", "--login" ]
