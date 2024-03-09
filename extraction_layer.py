"""Extraction Pipeline class"""

import re
import json


class ExtractionLayer:

    def __init__(self) -> None:
        pass

    
    def preprocess(file_name="complete_tex.tex"):

        with open(file_name,'r') as f:
            latex_string = f.readlines()

        figure_list = []
        Section_headings = {}
        for count,lines in enumerate(latex_string):
            if re.match(r'\section', lines):
                head = lines.lstrip(r'\section{').rstrip().rstrip('}')
                Section_headings[head] = 0
            if re.search(r'\includegraphics', lines):
                try:
                    figure_list.append(lines.split(']{')[1].rstrip().rstrip('}'))
                except:
                    pass
        return latex_string, figure_list, Section_headings

    def get_message_prompt(self, latex_string, prompt_base):


        messages =[
            {
              "role": "system",
              "content": "You are an expert slide expert that extracts latex papers and create presentation slides.",
            },
            {
              "role": "user",
              "content": f'''
              Given the document: {latex_string}
              Analyse paper by section and strictly follow this template but json format:
              {prompt_base}
              '''
            }
        ]
        
        return messages
  

    def prompt_builder(self, title: str,page_length: int,count:int, choices: list):
        
        latex_string, figure_list, Section_headings = self.preprocess()

        prompt_builder = f'''
        {{Section Name:  {title}
        Slide Information: {{
          Number of Slides: {page_length}
          Speaker Notes: {{"Slide Number": Long and Detailed Speaker Notes}}
          Table: Latex Table to dictionary if present in {title}>
          Image Prompt: Propose a prompt for slide background
        }}
        }}
        '''
        user_input = choices.split(',')
        for count,key in enumerate(Section_headings):
            Section_headings[key] = user_input[count]

        prompt_base = ''
        for count,(key,val) in enumerate(Section_headings.items()):
            prompt_base+=prompt_builder(key,val,count+1)

        output = self.get_message_prompt(latex_string, prompt_base)

        return output


