import os
import time
from typing import Optional
import openai
import pandas as pd
import random
from tqdm import tqdm


# from IPython.display import Markdown
openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


if __name__ == "__main__":
    base_dir = "datasets/WEB360/"
    txt_name = "caption.txt"
    csv_name = "WEB360_360TF.csv"
    csv_path = os.path.join(base_dir, csv_name)
    txt_path = os.path.join(base_dir, txt_name)

    text = []
    with open(txt_path, "r") as f:
        while True:
            text_0 = f.readline().strip().split(",")[-1]
            if not text_0:
                break
            text_90 = f.readline().strip().split(",")[-1]
            text_m90 = f.readline().strip().split(",")[-1]
            text_180 = f.readline().strip().split(",")[-1]
            text.append([text_0,text_90,text_m90,text_180])

    for i in tqdm(range(0, len(text))):
        text_0 = text[i][0]
        text_90 = text[i][1]
        text_m90 = text[i][2]
        text_180 = text[i][3]
        question = [{"0": text_0, "90": text_90, "-90": text_m90, "180": text_180}]
        response = openai.ChatCompletion.create(
            engine="dragon",
            messages=[
                {"role": "system",
                "content": "You are an expert in the field of image processing and computer vision. You excel at manipulating and synthesizing images, combining multiple images or image segments into a cohesive whole to create more informative and complete images. This skill is highly valuable in many applications, such as satellite image processing, urban planning, drone image analysis, and is widely used in these domains."},
                {"role": "user", "content": f"""
                Please give me the summarization of provided captions of different views. Each view groups include four views captured from the center of a same scene. The horizontal rotation angle of captured views is 0 degree, 90 degree, -90 degree and 180 degree, where 0 degree denotes the view is captured straight ahead. Please strictly adhere to the provided format without additional explanations.

                For example, if the input is: [{{"0": "an aerial view of a building at night", "90": "an aerial view of a parking lot at night", "-90": "a night time view of a city with buildings and lights", "180": "an aerial view of a city street at night"}}]
                The output should be: {"an aerial view of a city at night, the city including buildings, lights, a parking lot and a street"}

                If the input is : [{{"0": "a blurry photo of a cat on a table", "90": "a person standing in a room with a rug on the floor", "-90": "a blurry photo of a dog in a room", "180": "a blurry photo of people in a restaurant"}}]
                The output should be: {"a blurry photo of a person standing in a restaurant with a rug on the floor, a dog in the room, a cat on a table"}

                My input is：{question}"
                """}
            ],
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        res = response.choices[0].message.content
        text_360 = res.replace("The output should be: ", "")
        print(text_360)
        df = pd.DataFrame({"videoid": [i], "name": [text_360]})
        df.to_csv(csv_path, mode='a', index=False, header = False)
        time.sleep(random.randint(2, 5))