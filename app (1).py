from huggingface_hub import InferenceClient
import gradio as gr

client = InferenceClient(
    "google/gemma-7b-it"
)

def format_prompt(message, history):
    prompt = ""
    if history:
        #<start_of_turn>userWhat is recession?<end_of_turn><start_of_turn>model
        for user_prompt, bot_response in history:
            prompt += f"<start_of_turn>user{user_prompt}<end_of_turn>"
            prompt += f"<start_of_turn>model{bot_response}"
    prompt += f"<start_of_turn>user{message}<end_of_turn><start_of_turn>model"
    return prompt

def generate(
    prompt, history, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):
    if not history:
        history = []
        hist_len=0
    if history:
        hist_len=len(history)
        print(hist_len)

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output
    return output


additional_inputs=[
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=512,
        minimum=0,
        maximum=1048,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

# Create a Chatbot object with the desired height
chatbot = gr.Chatbot(height=450,
                     layout="bubble")

with gr.Blocks() as demo:
    gr.HTML("<h1><center>🤖 Google-Gemma-7B-Chat 💬<h1><center>")
    gr.ChatInterface(
        generate,
        chatbot=chatbot,  # Use the created Chatbot object
        additional_inputs=additional_inputs,
        examples=[["What is the meaning of life?"], ["Tell me something about Mt Fuji."]],

    )

demo.queue().launch(debug=True)
