ARG PY_IMAGE=python:3.11-slim-bullseye
ARG NODE_IMAGE=node:20-alpine
ARG NGINX_IMAGE=nginx:1.27-alpine

FROM ${PY_IMAGE} AS backend

ARG DEBIAN_MIRROR=http://mirrors.aliyun.com/debian
ARG DEBIAN_SECURITY_MIRROR=http://mirrors.aliyun.com/debian-security
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PIP_TRUSTED_HOST=mirrors.aliyun.com
ARG TORCH_SPEC=torch==2.8.0
ARG TORCHVISION_SPEC=torchvision==0.23.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN set -eux; \
    sed -i "s|http://deb.debian.org/debian|${DEBIAN_MIRROR}|g" /etc/apt/sources.list; \
    sed -i "s|http://security.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" /etc/apt/sources.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      libgl1 \
      libglib2.0-0 \
    ; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
COPY backend/requirements.core.txt /app/backend/requirements.core.txt
COPY vendor/ /app/vendor/
RUN set -eux; \
    python -m pip install --upgrade pip; \
    pip install --no-cache-dir -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" -r /app/backend/requirements.core.txt; \
    if [ -f /app/vendor/segment-anything-main.zip ]; then \
      pip install --no-cache-dir /app/vendor/segment-anything-main.zip; \
    else \
      pip install --no-cache-dir "segment-anything @ https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip"; \
    fi; \
    pip install --no-cache-dir -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" --extra-index-url "${TORCH_INDEX_URL}" "${TORCH_SPEC}" "${TORCHVISION_SPEC}"

COPY . /app

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM ${NODE_IMAGE} AS frontend-build

ARG PNPM_VERSION=10.17.0

WORKDIR /src/web

COPY web/package.json web/pnpm-lock.yaml /src/web/
RUN corepack enable && corepack prepare pnpm@${PNPM_VERSION} --activate && pnpm install --frozen-lockfile

COPY web/ /src/web/
RUN VITE_USE_MOCK=false pnpm build


FROM ${NGINX_IMAGE} AS frontend

COPY --from=frontend-build /src/web/dist /usr/share/nginx/html
COPY deploy/nginx/default.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
