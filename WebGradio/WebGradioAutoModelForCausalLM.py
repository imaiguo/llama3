
import os
import platform
import gradio as gr

from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

OpenAiServer=f"{BindLocalIP}:{OpenAiPort}"


if platform.system() == 'Windows':
    os.environ['PATH'] = os.environ.get("PATH", "") + os.pathsep + r'D:\devtools\PythonVenv\chatglb3\Lib\site-packages\torch\lib'

MODEL_PATH= "/opt/Data/ModelWeight/meta/llama3.hf/Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

def predict(chatbot, history):
    input = chatbot[-1][0]
    logger.debug(f"input->:{input}")
    
    jsoninput = {"role": "user", "content": input}
    history.append(jsoninput)
    logger.debug(f"input->history:{history}")
    input_ids = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda:0")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=8192,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    data = tokenizer.decode(response, skip_special_tokens=True)
    logger.debug(f"output->:{data}")

    jsoninput = {"role": "assistent", "content": data}
    history.append(jsoninput)
    logger.debug(f"history->:{history}")
    input = chatbot[-1][1] = data
    return chatbot, history

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], [], None

def add_text(chatbot, text):
    chatbot = chatbot + [(text, None)]
    return chatbot, gr.Textbox(value="", interactive=False)

with gr.Blocks(title = "智能客服小蓝", css="footer {visibility: hidden}").queue() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        height=800,
        show_copy_button = False,
        layout= "bubble",
        # avatar_images=((os.path.join(os.path.dirname(__file__), "../image/Einstein.jpg")), (os.path.join(os.path.dirname(__file__), "../image/openai.png")))
        avatar_images=("./image/Einstein.jpg", "./image/openai.png")
    )

    with gr.Row():
        with gr.Column(scale=9):
            user_input = gr.Textbox(show_label=False, placeholder="请输入您的问题,刷新页面可清除历史", lines=1, container=False)

        with gr.Column(min_width=1, scale=1):
            submitBtn = gr.Button("提交", variant="primary")

    begin = [{"role":"system", "content":"你是智能客服小蓝，仔细分析用户的输入，并作详细又准确的回答，记住使用中文回答问题。"}]
    history = gr.State(value=begin)

    subMsg = submitBtn.click(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True)
    inputMsg = user_input.submit(fn=add_text, inputs=[chatbot, user_input], outputs=[chatbot, user_input], queue=False).then(fn=predict, inputs=[chatbot, history], outputs=[chatbot, history], show_progress=True)

    subMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)
    inputMsg.then(fn=lambda: gr.Textbox(interactive=True), inputs=None, outputs=[user_input], queue=False)

auth=[(GradioUser,GradioPassword)]

demo.launch(server_name=BindLocalIP, server_port=int(BindPort), inbrowser=False, share=False, auth=auth)