#  MindSearch CPU-only 版部署

之前在彩蛋岛的任务中用到了DuckDuckGoSearch 搜索引擎，这个需要魔法才能访问，很多小伙伴卡在网络搭建部署这块儿，这节内容使用 HuggingFace space +硅基流动的 API Key的方式部署。本次任务带大家一起完成HuggingFace space 这种部署方式。感兴趣之前搭建的方式可能需要找找开源的魔法，可参考我之前的文章，https://github.com/Dstarjohn/sspu-LLM-MIndSearch

HuggingFace space官方体验：[体验地址](https://huggingface.co/spaces/SmartFlowAI/MindSearch_X_SiliconFlow)

# 部署流程

## 获取 API Key

部署的第一步就是获取硅基流动的 API Key 啦。

首先，我们打开 https://account.siliconflow.cn/login 来注册硅基流动的账号（如果注册过，则直接登录即可）

在完成注册后，打开 https://cloud.siliconflow.cn/account/ak 来准备 API Key。首先创建新 API 密钥，然后点击密钥进行复制，以备后续使用。

![](./image/2.png)

## github

我们可以直接使用github土工的一个web版本的vscode来下载MIndSearch的相关代码，地址：[CodeSpace](https://github.com/codespaces)，进入Codespaces 浏览器会自动在新的页面打开一个web版的vscode.

我们新建一个目录用于存放 MindSearch 的相关代码，并把 MindSearch 仓库 clone 下来。在终端中运行下面的命令：

```python
mkdir -p /workspaces/mindsearch
cd /workspaces/mindsearch
git clone https://github.com/InternLM/MindSearch.git
cd MindSearch && git checkout b832275 && cd ..

# 创建环境
conda create -n mindsearch python=3.10 -y
# 激活环境
conda activate mindsearch
source activate
# 安装依赖
pip install -r /workspaces/mindsearch/MindSearch/requirements.txt
# 硅基流动 API 的相关配置已经集成在了 MindSearch 中，所以我们可以直接执行下面的代码来启动 MindSearch 的后端——启动MIndSearch的后端

# 硅基流动 API Key。这里用自己创建的key，我这里就不复制出来了
export SILICON_API_KEY=
conda activate mindsearch
cd /workspaces/mindsearch/MindSearch
python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine DuckDuckGoSearch

# 接下来启动前端服务
conda activate mindsearch
cd /workspaces/mindsearch/MindSearch
python frontend/mindsearch_gradio.py

```

![](.\image\3.png)

打开gradio 自动创建的二级域名[URL 地址](https://6af4cab6a65b06dec6.gradio.live/)，就可以体验了

![](.\image\1.png)

## 部署到 HuggingFace Space

首先进入到[huggface_spaces](https://huggingface.co/spaces) ，并点击 Create new Space，如下图所示

![](.\image\10.png)

![](.\image\11.png)



![](.\image\5.png)

然后直接下拉到下面，选择 New secrets，name 一栏输入 SILICON_API_KEY，value 一栏输入你的 API Key 的内容

![](.\image\6.png)

**这里特别注意，我们的Name设置为SILICON_API_KEY，下面的key就是硅基流动的那个key，直接cv过来即可，这里硅基流动 API 的相关配置已经集成在了 MindSearch 中，所以我们不需要再去修改 `/path/to/MindSearch/mindsearch/models.py` 加上调用硅基流动 API 的相关配置了**

![](.\image\7.png)



接下来我们开始写代码了，创建目录把我们准备提交到Huggingface的代码文件准备好。

```python
# 创建新目录
mkdir -p /workspaces/mindsearch/mindsearch_deploy
# 准备复制文件
cd /workspaces/mindsearch
cp -r /workspaces/mindsearch/MindSearch/mindsearch /workspaces/mindsearch/mindsearch_deploy
cp /workspaces/mindsearch/MindSearch/requirements.txt /workspaces/mindsearch/mindsearch_deploy
# 创建 app.py 作为程序入口
touch /workspaces/mindsearch/mindsearch_deploy/app.py

```



app.py的内容如下：

```python
import json
import os

import gradio as gr
import requests
from lagent.schema import AgentStatusCode

os.system("python -m mindsearch.app --lang cn --model_format internlm_silicon &")

PLANNER_HISTORY = []
SEARCHER_HISTORY = []


def rst_mem(history_planner: list, history_searcher: list):
    '''
    Reset the chatbot memory.
    '''
    history_planner = []
    history_searcher = []
    if PLANNER_HISTORY:
        PLANNER_HISTORY.clear()
    return history_planner, history_searcher


def format_response(gr_history, agent_return):
    if agent_return['state'] in [
            AgentStatusCode.STREAM_ING, AgentStatusCode.ANSWER_ING
    ]:
        gr_history[-1][1] = agent_return['response']
    elif agent_return['state'] == AgentStatusCode.PLUGIN_START:
        thought = gr_history[-1][1].split('```')[0]
        if agent_return['response'].startswith('```'):
            gr_history[-1][1] = thought + '\n' + agent_return['response']
    elif agent_return['state'] == AgentStatusCode.PLUGIN_END:
        thought = gr_history[-1][1].split('```')[0]
        if isinstance(agent_return['response'], dict):
            gr_history[-1][
                1] = thought + '\n' + f'```json\n{json.dumps(agent_return["response"], ensure_ascii=False, indent=4)}\n```'  # noqa: E501
    elif agent_return['state'] == AgentStatusCode.PLUGIN_RETURN:
        assert agent_return['inner_steps'][-1]['role'] == 'environment'
        item = agent_return['inner_steps'][-1]
        gr_history.append([
            None,
            f"```json\n{json.dumps(item['content'], ensure_ascii=False, indent=4)}\n```"
        ])
        gr_history.append([None, ''])
    return


def predict(history_planner, history_searcher):

    def streaming(raw_response):
        for chunk in raw_response.iter_lines(chunk_size=8192,
                                             decode_unicode=False,
                                             delimiter=b'\n'):
            if chunk:
                decoded = chunk.decode('utf-8')
                if decoded == '\r':
                    continue
                if decoded[:6] == 'data: ':
                    decoded = decoded[6:]
                elif decoded.startswith(': ping - '):
                    continue
                response = json.loads(decoded)
                yield (response['response'], response['current_node'])

    global PLANNER_HISTORY
    PLANNER_HISTORY.append(dict(role='user', content=history_planner[-1][0]))
    new_search_turn = True

    url = 'http://localhost:8002/solve'
    headers = {'Content-Type': 'application/json'}
    data = {'inputs': PLANNER_HISTORY}
    raw_response = requests.post(url,
                                 headers=headers,
                                 data=json.dumps(data),
                                 timeout=20,
                                 stream=True)

    for resp in streaming(raw_response):
        agent_return, node_name = resp
        if node_name:
            if node_name in ['root', 'response']:
                continue
            agent_return = agent_return['nodes'][node_name]['detail']
            if new_search_turn:
                history_searcher.append([agent_return['content'], ''])
                new_search_turn = False
            format_response(history_searcher, agent_return)
            if agent_return['state'] == AgentStatusCode.END:
                new_search_turn = True
            yield history_planner, history_searcher
        else:
            new_search_turn = True
            format_response(history_planner, agent_return)
            if agent_return['state'] == AgentStatusCode.END:
                PLANNER_HISTORY = agent_return['inner_steps']
            yield history_planner, history_searcher
    return history_planner, history_searcher


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">MindSearch Gradio Demo</h1>""")
    gr.HTML("""<p style="text-align: center; font-family: Arial, sans-serif;">MindSearch is an open-source AI Search Engine Framework with Perplexity.ai Pro performance. You can deploy your own Perplexity.ai-style search engine using either closed-source LLMs (GPT, Claude) or open-source LLMs (InternLM2.5-7b-chat).</p>""")
    gr.HTML("""
    <div style="text-align: center; font-size: 16px;">
        <a href="https://github.com/InternLM/MindSearch" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">🔗 GitHub</a>
        <a href="https://arxiv.org/abs/2407.20183" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">📄 Arxiv</a>
        <a href="https://huggingface.co/papers/2407.20183" style="margin-right: 15px; text-decoration: none; color: #4A90E2;">📚 Hugging Face Papers</a>
        <a href="https://huggingface.co/spaces/internlm/MindSearch" style="text-decoration: none; color: #4A90E2;">🤗 Hugging Face Demo</a>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Row():
                with gr.Column():
                    planner = gr.Chatbot(label='planner',
                                         height=700,
                                         show_label=True,
                                         show_copy_button=True,
                                         bubble_full_width=False,
                                         render_markdown=True)
                with gr.Column():
                    searcher = gr.Chatbot(label='searcher',
                                          height=700,
                                          show_label=True,
                                          show_copy_button=True,
                                          bubble_full_width=False,
                                          render_markdown=True)
            with gr.Row():
                user_input = gr.Textbox(show_label=False,
                                        placeholder='帮我搜索一下 InternLM 开源体系',
                                        lines=5,
                                        container=False)
            with gr.Row():
                with gr.Column(scale=2):
                    submitBtn = gr.Button('Submit')
                with gr.Column(scale=1, min_width=20):
                    emptyBtn = gr.Button('Clear History')

    def user(query, history):
        return '', history + [[query, '']]

    submitBtn.click(user, [user_input, planner], [user_input, planner],
                    queue=False).then(predict, [planner, searcher],
                                      [planner, searcher])
    emptyBtn.click(rst_mem, [planner, searcher], [planner, searcher],
                   queue=False)

demo.queue()
demo.launch(server_name='0.0.0.0',
            server_port=7860,
            inbrowser=True,
            share=True)

```



回到刚才的github上面的CodeSpace 进到终端，把我们Huggingface上面的仓库下载下来

```python
cd /workspaces/codespaces-blank
# 下面的代码注意不能直接用把xxxx 换成你的token;  huggingface.co/spaces/dstars/mindsearch  换成你的项目URL
git clone https://dstars:xxxx@huggingface.co/spaces/dstars/mindsearch

# 这里我们需要先设置好远程仓库的URL
git remote set-url origin https://huggingface.co/spaces/dstars/MIndSearch
# 解下把我们的文件cp到指定路径下
cd /workspaces/mindsearch/mindsearch_deploy
cp app.py /workspaces/codespaces-blank/MIndSearch
cp requirements.txt /workspaces/codespaces-blank/MIndSearch
cp -r mindsearch/  /workspaces/codespaces-blank/MIndSearch
# 后面就是正常的git提交的流程了
cd /workspaces/codespaces-blank/mindsearch
git init
git add .
git commit -m "Add application file"
git push origin
# 如果提交失败的话，多半是因为Token令牌没设置或者是权限的问题
```



如果是权限问题记得创建这个token令牌并且开放写入权限。

![](.\image\8.png)

![](.\image\13.png)

这是push后的仓库文件

![](.\image\12.png)



然后皆可以愉快的访问啦

![](.\image\9.png)

部署体验地址：[**LX_HF**的MIndSearch](https://huggingface.co/spaces/dstars/MIndSearch)
