import openai
import anthropic
import os
import json
import warnings
import datetime
from PIL import Image
import io
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Set your Stability API key from environment variables for better security practices
os.environ["STABILITY_KEY"] = os.getenv("STABILITY_KEY", "")


class DataLayer:
    def __init__(self, save_folder="output"):
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_key = os.getenv(
            "OPENAI_API_KEY", ""
        )  # It's a good practice to load keys from environment variables
        self.stability_api = client.StabilityInference(
            key=os.getenv("STABILITY_KEY", ""),  # API Key from environment variable
            verbose=True,
            engine="stable-diffusion-xl-1024-v1-0",
        )
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder

    def bullet_points(self, prompt):
        output = self.get_completion_from_messages(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert presentation designer that create bullet points from speech",
                },
                {
                    "role": "user",
                    "content": f"Given speech of: {prompt}\nGive me 4 concise, 10 words bullet points for the slide",
                },
            ],
            model="gpt-3.5-turbo",
        )
        bullets = [x.strip("* ").strip() for x in output.split("\n")]
        return bullets

    def get_completion_from_messages(
        self, messages, model="gpt-4-turbo-preview", openAI=True, temperature=0, max_tokens=4000
    ):
        if openAI:
            openai.api_key = self.openai_key
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=self.anthropic_key,
            )
            response = client.messages.create(
                model=model,  # "claude-3-opus-20240229"
                max_tokens=max_tokens,
                temperature=temperature,
                system=messages[0]["content"],
                messages=messages[1:],
            )
        return response.choices[0].message.content if openAI else response.content[0].text

    def llm_output_to_dict(self, llm_output, section_name, slide_number):
        llm_output = llm_output.lstrip("```json").rstrip("```").strip()
        json_output = json.loads(llm_output)
        return {
            "Paper Title": json_output["Paper Title"],
            "section_name": section_name,
            "slide_number": slide_number,
            "slide_title": json_output[section_name]["slides_information"][slide_number - 1]["slide_title"],
            "speaker_notes": json_output[section_name]["slides_information"][slide_number - 1]["speaker_notes"],
            "table": json_output[section_name]["slides_information"][slide_number - 1]["table"],
            "image": json_output[section_name]["slides_information"][slide_number - 1]["image"],
            "generative_prompt": json_output[section_name]["slides_information"][slide_number - 1]["generative_prompt"],
        }

    def generate_image(self, slide_image="", generative_prompt=""):
        Time_hash = datetime.datetime.today().strftime(r"%H%M%S")
        if slide_image == "":
            print("No image in this section, Stable Diffusion called")
            answers = self.stability_api.generate(
                prompt=generative_prompt,
                seed=4253978046,
                steps=50,
                cfg_scale=8.0,
                width=1024,
                height=1024,
                samples=1,
                sampler=generation.SAMPLER_K_DPMPP_2M,
            )

            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        warnings.warn(
                            "Your request activated the API's safety filters and could not be processed. "
                            "Please modify the prompt and try again."
                        )
                    elif artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        generated_image = os.path.join(self.save_folder, f"{Time_hash}_{artifact.seed}.png")
                        img.save(generated_image)
                        return generated_image
        else:
            print("Image Present in this section")
            return slide_image


if __name__ == "__main__":
    dl = DataLayer()
    llm_output = dl.get_completion_from_messages(messages, openAI=False, model="claude-3-sonnet-20240229")
    print(llm_output)
    for x in el.section_headings:
        for y in range(1, el.section_headings[x] + 1):
            per_slide_dict = dl.llm_output_to_dict(llm_output, x, y)
            # print(per_slide_dict)
            per_slide_bullets = dl.bullet_points(per_slide_dict["speaker_notes"])
            # print(per_slide_bullets)
            per_slide_dict["image"] = dl.generate_image(per_slide_dict["image"], per_slide_dict["generative_prompt"])
            # print(per_slide_dict)
