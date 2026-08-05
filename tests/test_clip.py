from random import choice

from transformers import AutoTokenizer, CLIPTextModel, CLIPVisionModel, AutoProcessor

import torch


def test_text():
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


def test_image():
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    img = torch.randn(1, 3, 96, 96)
    inputs = processor(images=img, return_tensors="pt")
    print(inputs.pixel_values.shape)
    outputs = model(**inputs)
    pooled_output = outputs.pooler_output

    print(pooled_output.shape)


if __name__ == "__main__":
    test_image()
