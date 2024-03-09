import openai
import os
import json


class DataLayer:

    def __init__(self) -> None:
        self.openai_key = ''


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

