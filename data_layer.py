import openai
import os
import json
import io
import warnings
from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

os.environ['STABILITY_KEY'] = "sk-cBT0Gn4z1YYcqgR4k3KuDbw0Jx7xrzc5hT215JkFUIQHVo3A"


class DataLayer:

    def __init__(self, slide_image, generative_prompt) -> None:
        self.openai_key = '',
        self.stability_api = client.StabilityInference(
            key='', # API Key reference.
            verbose=True, # Print debug messages.
            engine="stable-diffusion-xl-1024-v1-0", # Set the engine to use for generation
        )
    
        self.my_dict = {
            f'Title':
            {
                'Information':'Some info',
                'Page Number':'Some info',
                'Images':'Some Image',
                'Tables':'Some Table'
                }
            }
        
        self.slide_image = slide_image,
        self.generative_prompt = generative_prompt,

    def bullet_points(self, prompt):
        bullets = []
        for info in prompt:
            output = self.get_completion_from_messages(messages=[
                {
                "role": "system",
                "content": "You are an expert presentation designer that create bullet points from speech",
                },
                {
                "role": "user",
                "content": f'''Given the document: {info}
                Give me 4 concise, 10 words bullet points for the slide''',
                }
            ],model='gpt-3.5-turbo',)
            bullets.append([x.lstrip('* ').rstrip() for x in output.split('\n')])
        return bullets
    

    def get_completion_from_messages(self, messages,
                                 model="gpt-4-turbo-preview",
                                 response_format={ "type": "json_object" },
                                 temperature=0, max_tokens=1200):
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


    def llm_output_to_dict(self, llm_output, content_num):
  
        section_json = []
        for x in llm_output.split('\n\n'):
            section_json.append(x.replace('```','').lstrip().rstrip().lstrip('json'))
        section_json = [x for x in section_json if not x == '']
        result = json.loads(f'{section_json[content_num]}')
        section_name = result['Section Name']
        number_of_slides = result['Slide Information']['Number of Slides']
        slide_info = [x for x in result['Slide Information']['Speaker Notes'].values()]
        table = result['Slide Information']['Table']
        image = result['Slide Information']['Image']
        generative_prompt = result['Slide Information']['Generative Prompt']
        return section_name,number_of_slides,slide_info,table,image,generative_prompt


    def generate_image(self):
        
        if self.slide_image == None:
            print('No image in this section, Stable Diffusion called')
            answers = self.stability_api.generate(
            prompt= self.generative_prompt,
            seed=4253978046,
            steps=50, # Amount of inference steps performed on image generation. Defaults to 30.
            cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                            
            width=1024, # Generation width, defaults to 512 if not included.
            height=1024, # Generation height, defaults to 512 if not included.
            samples=1, # Number of images to generate, defaults to 1 if not included.
            sampler=generation.SAMPLER_K_DPMPP_2M 
                                                 
            )

            for resp in answers:
                for artifact in resp.artifacts:
                    if artifact.finish_reason == generation.FILTER:
                        warnings.warn(
                            "Your request activated the API's safety filters and could not be processed."
                            "Please modify the prompt and try again.")
                    if artifact.type == generation.ARTIFACT_IMAGE:
                        img = Image.open(io.BytesIO(artifact.binary))
                        slide_image = str(artifact.seed)+ ".png"
                        img.save(slide_image)
        else:
            print('Image Present in this section')
            pass