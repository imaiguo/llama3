
import os
import gradio as gr

from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BindLocalIP = os.getenv('LocalIP')
BindPort = os.getenv('BindPort')
OpenAiPort = os.getenv('OpenAiPort')
GradioUser = os.getenv('GradioUser')
GradioPassword = os.getenv('GradioPassword')

print(f"BindLocalIP: {BindLocalIP}")
print(f"BindPort: {BindPort}")
print(f"OpenAiPort: {OpenAiPort}")
print(f"GradioUser: {GradioUser}")
print(f"GradioPassword: {GradioPassword}")

OpenAiServer="localhost:11434"

client = OpenAI(base_url=f"http://{OpenAiServer}/v1", api_key = "not need key")

def predict(chatbot, history):
    input = chatbot[-1][0]
    logger.debug(f"input->:{input}")
    history += [{"role": "user", "content": input}]
  
    response = client.chat.completions.create(model='llama3:8b-instruct-fp16',
                                              messages= history,
                                              temperature=1.0,
                                              max_tokens=32768,
                                              stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            chatbot[-1][1] = partial_message
            yield chatbot, history

    history += [{"role": "assistant", "content": partial_message}]
    logger.debug(f"history: {history}")

def add_text(history, text):
    history = history + [(text, None)]
    return history, gr.Textbox(value="", interactive=False)

with gr.Blocks(title = "智能客服小蓝", css="footer {visibility: hidden}") as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=800,
        show_copy_button = False,
        layout= "bubble",
        avatar_images=("./image/Einstein.jpg", "./image/openai.png")
    )

    with gr.Row():
        with gr.Column(scale=9):
            user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题,刷新页面可清除历史", lines=1, container=False)

        with gr.Column(min_width=1, scale=1):
            submitBtn = gr.Button("提交", variant="primary")

    begin = [{"role":"system", "content":"你是智能客服小蓝，仔细分析用户的输入，并作详细又准确的回答，记住使用中文回答问题。"}]
    history = gr.State(value=begin)

    subMsg = submitBtn.click(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=3)
    inputMsg = user_input.submit(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True, concurrency_limit=3)

    subMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)
    inputMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)

auth=[(GradioUser, GradioPassword)]

demo.queue().launch(server_name=BindLocalIP, server_port=int(BindPort), inbrowser=False, share=False, auth=auth)
