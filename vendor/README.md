# vendor 依赖说明

为避免 Docker build 时每次从 GitHub 拉取 `segment-anything` 导致超时，可把源码 zip 包预先放到本目录。

## 使用方式

1. 下载 zip 包：
   - https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip
2. 保存为：
   - `vendor/segment-anything-main.zip`
3. 重新构建镜像。

Dockerfile 会优先从本地 `vendor/segment-anything-main.zip` 安装；如果该文件不存在，才会退回到在线下载。

