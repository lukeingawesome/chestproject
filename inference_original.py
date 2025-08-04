from transformers import pipeline
from PIL import Image
import requests
import torch
import pandas as pd

df = pd.read_csv('/data3/private/snu/supplementary/snu_test_outputs.csv')
import os
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

from tqdm import tqdm

for i in tqdm(range(len(df))):
    img_path = df.loc[i, 'preprocessed_path']
    image = Image.open(img_path)
    

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Generate the findings of the following X-ray image."},
                {"type": "image", "image": image}
            ]
        }
    ]
    output = pipe(text=messages, max_new_tokens=300)
    df.loc[i, 'medgemma'] = output[0]["generated_text"][-1]["content"]

df.to_csv('snu_test_outputs_original.csv', index=False)
