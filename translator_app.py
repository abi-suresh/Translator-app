from transformers import MarianMTModel, MarianTokenizer
import gradio as gr

# Load the English → French translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translation function
def translate(text):
    if not text.strip():
        return ""
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# App Interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    <h1 style="text-align: center;">🌍 AI Translator: English → French</h1>
    <p style="text-align: center;">Translate English text into French .</p>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="✍️ English Text",
                placeholder="Type here...",
                lines=6,
                interactive=True
            )
            char_count = gr.Markdown()
            examples = gr.Examples(
                examples=[
                    ["Good morning!"],
                    ["How are you today?"],
                    ["I love machine learning."],
                ],
                inputs=input_text
            )

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="🇫🇷 Translated French",
                lines=6,
                interactive=False
            )
            copy_btn = gr.Button("📋 Copy Translation")

    # Buttons and Logic
    input_text.change(fn=translate, inputs=input_text, outputs=output_text)
    input_text.change(lambda txt: f"**Characters:** {len(txt)}", inputs=input_text, outputs=char_count)
    copy_btn.click(lambda text: text, inputs=output_text, outputs=None)

    with gr.Row():
        clear_btn = gr.Button("🧹 Clear")
        clear_btn.click(fn=lambda: ("", "", "**Characters:** 0"), inputs=None, outputs=[input_text, output_text, char_count])

# Run
if __name__ == "__main__":
    app.launch()
