import openai
import os
import json
import warnings
from PIL import Image
import io
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# Set your Stability API key from environment variables for better security practices
os.environ["STABILITY_KEY"] = os.getenv("STABILITY_KEY", "your_default_key")


class DataLayer:
    def __init__(self, slide_image=None, generative_prompt=""):
        # It's a good practice to load keys from environment variables
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.stability_api = client.StabilityInference(
            key=os.getenv("STABILITY_KEY", ""),  # API Key from environment variable
            verbose=True,
            engine="stable-diffusion-xl-1024-v1-0",
        )

        self.slide_image = slide_image
        self.generative_prompt = generative_prompt

    def bullet_points(self, prompt):
        bullets = []
        for info in prompt:
            output = self.get_completion_from_messages(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert presentation designer that create bullet points from speech",
                    },
                    {
                        "role": "user",
                        "content": f"Given the document: {info}\nGive me 4 concise, 10 words bullet points for the slide",
                    },
                ],
                model="gpt-3.5-turbo",
            )
            bullets.append([x.strip("* ").strip() for x in output.split("\n")])
        return bullets

    def get_completion_from_messages(self, messages, model="gpt-4-turbo-preview", temperature=0, max_tokens=1300):
        openai.api_key = self.openai_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def llm_output_to_dict(self, llm_output, content_num):
        Section_Json = []
        for x in llm_output.split("\n\n"):
            Section_Json.append(x.replace("```", "").lstrip().rstrip().lstrip("json"))
        Section_Json = [x for x in Section_Json if not x == ""]
        if Section_Json:
            result = json.loads(f"{Section_Json[content_num]}")
            return {
                "section_name": result["Section Name"],
                "number_of_slides": result["Slide Information"]["Number of Slides"],
                "slide_info": list(result["Slide Information"]["Speaker Notes"].values()),
                "table": result["Slide Information"]["Table"],
                "image": result["Slide Information"]["Image"],
                "generative_prompt": result["Slide Information"]["Generative Prompt"],
            }
        else:
            return {}

    def generate_image(self):
        if not self.slide_image:
            print("No image in this section, Stable Diffusion called")
            answers = self.stability_api.generate(
                prompt=self.generative_prompt,
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
                            "Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again."
                        )
                    elif artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        slide_image = f"{artifact.seed}.png"
                        img.save(slide_image)
        else:
            print("Image Present in this section")