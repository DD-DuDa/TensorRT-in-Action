# TensorRT-in-Action
TensorRT-in-Action 是一个 GitHub 代码库，提供了使用 TensorRT 的代码示例，并有对应 Jupyter Notebook。 

## TensorRT 安装
**推荐采用 Docker 安装**

在下面的网站，可以直接安装带有最新 TensorRT, Pytorch, cuDNN, NCCL 等软件的 docker 镜像

[PyTorch | NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

若使用远程服务器，请使用如下命令登录：

```bash
ssh -L localhost:8900:localhost:8900 {name}@{ip}
```

请（在你的服务器上）运行下面的命令：

```bash
docker run -it --gpus all -p 8900:8900 --name trt-action --shm-size 32G --ulimit memlock=-1 --ulimit stack=67108864 -v ~:/work [nvcr.io/nvidia/pytorch:23.04-py3](http://nvcr.io/nvidia/pytorch:23.04-py3) /bin/bash
```

- `docker run`：运行 Docker 容器的命令
- `-it`：为容器分配一个伪终端，并保持 STDIN 处于打开状态，即使未连接也是如此
- `--gpus all`：使容器内的程序可以访问主机上的所有 GPU
- `-p 8900:8900`：将容器内的端口8900映射到主机上的端口8900。这允许在主机上通过指定的端口访问容器内运行的服务或应用程序。
- `--name trt-action`：将容器命名为 "trt-action"
- `--shm-size 32G`：设置容器的共享内存大小为32GB。共享内存用于在容器内的进程之间共享数据。
- `--ulimit memlock=-1`：设置内存锁定的限制。将值设置为-1表示不限制内存锁定，允许进程锁定任意数量的内存。
- `--ulimit stack=67108864`：设置堆栈大小的限制。这里将堆栈大小限制为67108864字节（64MB）。
- `-v ~:/work`：将用户的主目录挂载为容器内的 /work 目录
- `[nvcr.io/nvidia/pytorch:23.04-py3](<http://nvcr.io/nvidia/pytorch:23.04-py3>)`：指定要运行的 Docker 镜像，即安装了 TensorRT 8.6 的 NVIDIA PyTorch 镜像
- `/bin/bash`：在容器内执行**`/bin/bash`**命令，以便在容器中启动一个交互式的Bash终端会话。这将成为容器的入口点。

若是第一次运行该命令，将会自动下载 Docker 镜像

**常用指令**
```bash
# 开启容器
docker start trt-action
# 进入容器命令窗口
docker exec -it trt-action /bin/bash
```