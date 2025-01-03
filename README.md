
<div align="center">
    <h1>Just Api For Comfy Workflow</h1>
</div>

#### 项目结构 （Project structure）
```
comfy-api
    ├── comfy                            # comfy核心代码目录
    ├── third_nodes                     # 三方节点代码目录
    ├── workflows                        # 工作流代码目录
    ├── app                              # 应用代码目录
    ├── models                           # 模型文件目录
    ├── inputs                           # 输入文件目录
    ├── outputs                          # 输出文件目录
    ├── test                             # 测试目录
    ├── .gitignore                       # git忽略文件
    ├── README.md                        # 描述文件
    └── requirements.txt                 # 依赖包
    └── Dockerfile                       # Dockerfile
    └── main.py                          # 程序入口
    └── folder_paths.py                  # 路径相关配置
    └── utils.py                         # 工具类
    └── server.py                        # WEB服务
    └── nodes.py                         # 节点类
    └── node_helpers.py                  # 节点辅助类

```

#### 描述（Description）

构建独立的ComfyUI工作流API服务

Build standalone ComfyUI workflow API service

1、为什么要实现这个项目？

Why do I need to implement this project?

答：因为ComfyUI的UI交互体验太差，无法满足我的应用产品需求。

Answer: Because the UI interaction experience of ComfyUI is too poor, which cannot meet my application product requirements.

2、这个项目有什么作用？

What is the purpose of this project?

答：这个项目是一个独立的ComfyUI工作流的代码实现，利用异步队列和线程，处理客户端的请求，实现用户根据参数执行生图工作流，并将最终结果以接口响应形式返回给客户端渲染。

Answer: This project is a standalone implementation of a ComfyUI workflow, using asynchronous queues and threads to handle client requests, implement user execution of the generation workflow based on parameters, and return the final results in interface response form to the client for rendering.

3、这个项目有什么意义？

What is the significance of this project?

答：这个项目具有以下意义：
- 面向程序开发者，推动AI应用项目落地，而不是停留在工作流阶段。
- 通过使用本项目，并组织编写相关工作流代码，可以更深入理解torch、大模型、SD等技术原理。
- 实现部署更方便。
- 更轻量级的代码，避免具体应用落地时，包含太多不需要的节点和依赖。
- 可扩展性更强，可以高度自定义工作流过程。

Answer: This project has the following meanings:
- Oriented to programmers, pushing AI application projects to the ground and not staying at the workflow stage.
- By using this project and organizing the writing of workflow code, you can better understand the principles of torch, large models, and SD.
- Implementation deployment is easier.
- A lighter code, avoiding the inclusion of too many unnecessary nodes and dependencies when specific application deployment.
- Extensibility is stronger, you can customize the workflow process more flexibly.

#### 快速开始（Quick Start）

0、拉取代码： `git clone https://github.com/darifo/comfy-api.git`

进入项目目录： `cd comfy-api`

创建虚拟环境： `python -m venv myenv` 

使用conda创建虚拟环境： `conda create -n myenv python=3.12`

激活虚拟环境： `source myenv/bin/activate` or `conda activate myenv`

1、安装依赖包

```shell
pip install -r requirements.txt
```

2、启动服务

```shell
python main.py
```

3、访问服务

```shell
http://localhost:9609
```

#### 工作流实现（Workflow Implementation）

创建工作流代码文件：`workflows/my_workflow.py`

#### 在应用中使用工作流（Use Workflow in Application）







