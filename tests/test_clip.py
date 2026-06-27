from transformers import AutoTokenizer, CLIPTextModel
from random import choice

if __name__ == "__main__":
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokeniser = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    colors = ["red", "blue", "green"]

    def generate_input():
        return f"Pick {choice(colors)} color block and place it on the {choice(colors)} bin"

    text_inputs = [generate_input() for _ in range(32)]
    print(text_inputs)

    input = tokeniser(text_inputs, padding=True, return_tensors="pt")
    output = model(**input)
    pooled_output = output.pooler_output

    print(pooled_output.shape)
