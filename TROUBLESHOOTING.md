# Distributed Training Troubleshooting Guide

## Common Single-Node Issues & Fixes

| Root Cause | Symptom | Fix |
|------------|---------|-----|
| **Firewall/SELinux rejects loop-back TCP** | `ECONNABORTED` on `127.0.0.1:port` | `sudo ufw disable` or `sudo systemctl stop firewalld`<br/>On RHEL/CentOS: `sudo setenforce 0` |
| **Rootless Docker/Podman** | Server IP is `172.x.x.x` or connect to `127.0.0.1` fails | Run container with:<br/>`--network=host --ipc=host --ulimit memlock=-1:-1` |
| **Kernel keeps ports in TIME_WAIT** | Fails only on 2nd/3rd launch | `export NCCL_REUSE_SOCKET_ADDR=0` |
| **NCCL 2.21.5 bug (PyTorch 2.3)** | nccl-tests fail, PyTorch fails | Upgrade to nightly with NCCL 2.21.7+:<br/>`pip install --pre torch==2.4.0.dev*.cu121 --index-url https://download.pytorch.org/whl/nightly/cu121` |
| **Per-user open-file limit too low** | Random ranks abort, ENOMEM in dmesg | `ulimit -n 1048576` before launching |

## Environment Variables Reference

### Required for Single-Node Multi-GPU:
```bash
export NCCL_SOCKET_IFNAME=lo              # Use loopback only
export NCCL_IB_DISABLE=1                  # Disable InfiniBand
export TORCH_NCCL_BLOCKING_WAIT=1         # PyTorch 2.3+
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # PyTorch 2.3+
export NCCL_REUSE_SOCKET_ADDR=0           # Force fresh ports
```

### Optional (if training fails after successful ping):
```bash
export ACCELERATE_DISABLE_NCCL=1          # Use Gloo for Accelerate's first barrier
```

## What Success Looks Like

All four ranks should print their initialization:
```
[rank 0/4] pid 12345 gpu cuda:0 master 127.0.0.1:45234
[rank 1/4] pid 12346 gpu cuda:1 master 127.0.0.1:45234
[rank 2/4] pid 12347 gpu cuda:2 master 127.0.0.1:45234
[rank 3/4] pid 12348 gpu cuda:3 master 127.0.0.1:45234
```

## Testing NCCL

Run nccl-tests after each change:
```bash
cd /tmp
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda
./build/all_reduce_perf -b 8 -e 256M -f 2 -g 4
```

Should show bandwidth numbers, not abort.

## TIME_WAIT Handling

### Option 1: Wait 60 seconds between launches
TIME_WAIT expires naturally after 60 seconds.

### Option 2: Reduce kernel timeout (requires root on host)
```bash
sudo sysctl -w net.ipv4.tcp_fin_timeout=15     # default 60
# or enable safe reuse:
sudo sysctl -w net.ipv4.tcp_tw_reuse=1
```

### Option 3: Hard-code a safe port
```bash
export MASTER_PORT=29555
deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT train_medgemma.py
```
Wait 60s after crashes before relaunching.

## Docker Launch Command

For containerized training:
```bash
docker run -it --gpus all --network=host --ipc=host \
  --ulimit memlock=-1:-1 --shm-size 8g \
  --sysctl net.ipv4.tcp_tw_reuse=1 \
  your-image ./run_train.sh
```

**Note**: Container sysctl settings require host privileges. If you get "Read-only file system" errors, the sysctls must be set on the host or at container runtime with `--sysctl` flags.

## Summary Recipe (Eliminates Both Errors)

```bash
pkill -f train_medgemma.py deepspeed.launcher torchrun
sleep 3                                 # give kernel time to close sockets
ulimit -n 1048576

export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export ACCELERATE_DISABLE_NCCL=1        # bypass fragile barrier

# Choose a port that is not LISTEN *or* TIME_WAIT
while :; do
  PORT=$(( 40000 + RANDOM % 20000 ))
  ss -tan | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT
echo "Launching on port $PORT"

deepspeed --include localhost:0,1,2,3 --master_port $PORT train_medgemma.py
``` 