# Server Deployment

This project uses Docker Compose with an Nginx reverse proxy.

External port:

```bash
http://SERVER_IP:53378
```

## Directory layout on server

Keep these directories next to `docker-compose.yml`:

```text
models/   model files, mounted read-only
img/      gallery images and uploads
result/   generated outputs
configs/  model and weight config
```

The front gallery should be placed in:

```text
img/front/
```

## Start

CPU-only mode (default):

```bash
docker compose up -d --build
```

GPU mode:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The GPU mode requires NVIDIA Driver and NVIDIA Container Toolkit on the server.

The CPU-only mode (default) pins PyTorch to CPU wheels:

```text
torch==2.8.0
torchvision==0.23.0
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
```

The GPU mode pins PyTorch to CUDA 12.8 wheels:

```text
torch==2.8.0+cu128
torchvision==0.23.0+cu128
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
```

This matches hosts where `nvidia-smi` reports `CUDA Version: 12.8`, such as
NVIDIA driver `570.x`. If PyTorch reports `torch.version.cuda: 13.0` and
`torch.cuda.is_available(): False`, rebuild the image after pulling the latest
project files:

```bash
docker compose down
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build --no-cache similarity-backend
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

If container startup reports `failed to discover GPU vendor from CDI`, first verify the
host GPU runtime:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the Docker test fails, install or reconfigure NVIDIA Container Toolkit:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then start again:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

To start without GPU while fixing the runtime:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
docker compose up -d
```

## Build mirrors

The Docker build defaults to Aliyun mirrors:

- Debian APT: `http://mirrors.aliyun.com/debian`
- Debian security: `http://mirrors.aliyun.com/debian-security`
- PyPI: `https://mirrors.aliyun.com/pypi/simple`

To use Tsinghua mirrors instead:

```bash
DEBIAN_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian \
DEBIAN_SECURITY_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/debian-security \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

To use USTC mirrors:

```bash
DEBIAN_MIRROR=https://mirrors.ustc.edu.cn/debian \
DEBIAN_SECURITY_MIRROR=https://mirrors.ustc.edu.cn/debian-security \
PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple \
PIP_TRUSTED_HOST=mirrors.ustc.edu.cn \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

## Operations

```bash
docker compose ps
docker compose logs -f similarity-backend
docker compose restart
docker compose down
```

## Notes

- Nginx listens on host port `53378`, serves `web/` build output, and proxies `/api` + `/assets` to FastAPI (`similarity-backend:8000`).
- `models`, `img`, and `result` are not baked into the image. They are mounted from the host.
- Increase `client_max_body_size` in `deploy/nginx/default.conf` if uploads exceed 250 MB.
