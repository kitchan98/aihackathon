import re
import json

class ExtractionLayer:
    def __init__(self, file_name="complete_tex.tex"):
        self.file_name = file_name
        self.latex_string, self.figure_list, self.section_headings = self.preprocess()

    def preprocess(self):
        """Read LaTeX file and extract relevant information."""
        with open(self.file_name, 'r') as f:
            latex_string = f.readlines()

        figure_list = []
        section_headings = {}
        for count, line in enumerate(latex_string):
            if re.match(r'\\section', line):
                head = line.lstrip('\\section{').rstrip().rstrip('}')
                section_headings[head] = []
            if re.search(r'\\includegraphics', line):
                figure_path = self.extract_figure_path(line)
                if figure_path:
                    figure_list.append(figure_path)
        return latex_string, figure_list, section_headings

    @staticmethod
    def extract_figure_path(line):
        """Extract figure path from a line if present."""
        try:
            return line.split(']{')[1].rstrip().rstrip('}')
        except IndexError:
            return None

    def get_message_prompt(self, prompt_base):
        """Construct message prompts for interaction."""
        messages = [
            {
                "role": "system",
                "content": "You are an expert slide expert that extracts latex papers and create presentation slides.",
            },
            {
                "role": "user",
                "content": f'''
                Given the document: {self.latex_string}
                Analyse paper by section and strictly follow this template but in json format:
                {prompt_base}
                '''
            }
        ]
        return messages

    def prompt_builder(self, title: str, page_length: int, count: int, user_choices: str):
        ###Build the prompt based on extracted information and user input.
        user_input = user_choices.split(',')
        for count, key in enumerate(self.section_headings.keys()):
            self.section_headings[key] = user_input[count] if count < len(user_input) else ""

        prompt_base = ''
        for count, (key, val) in enumerate(self.section_headings.items(), start=1):
            prompt_base += self._format_section_prompt(title=key, page_length=page_length, count=count)

        return self.get_message_prompt(prompt_base)

    def _format_section_prompt(self, title, page_length):
        """Format section prompts based on titles and other parameters."""
        prompt_template = f'''
        Section Name: {title}
        Slide Information: {{
          Number of Slides: {page_length},
          Speaker Notes: {{"Slide {{Number}}": "150 words speech for slide"}},
          Table: Latex Table to header and row dictionary if present in {title},
          Image: Figure path if present in {title} section (possible selections: {self.figure_list}),
          Generative Prompt: Propose a cartoonist image prompt related to {title}
        }}
        '''
        return prompt_template
