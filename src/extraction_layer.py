"""This module contains the ExtractionLayer class that extracts information from a LaTeX file."""

import pprint
import re


class ExtractionLayer:
    def __init__(self, file_name="complete_tex.tex"):
        self.file_name = file_name
        self.latex_string, self.figure_list, self.section_headings = self.preprocess()

    def preprocess(self):
        """Read LaTeX file and extract relevant information."""
        with open(self.file_name, "r") as f:
            latex_string = f.readlines()

        figure_list = []
        section_headings = []
        for line in latex_string:
            if re.match(r"\\section", line):
                head = line.lstrip("\\section{").rstrip().rstrip("}")
                section_headings.append(head)
            if re.search(r"\\includegraphics", line):
                figure_path = self.extract_figure_path(line)
                if figure_path:
                    figure_list.append(figure_path)
        return latex_string, figure_list, section_headings

    @staticmethod
    def extract_figure_path(line):
        """Extract figure path from a line if present."""
        try:
            return line.split("]{")[1].rstrip().rstrip("}")
        except IndexError:
            return None

    def get_message_prompt(self, prompt_base):
        """Construct message prompts for interaction."""
        messages = [
            {
                "role": "system",
                "content": "You are a scientific expert that can extract relevant information from"
                " latex papers to assisnt another agent in creating presentation slides.",
            },
            {
                "role": "user",
                "content": f"""Given the document: <start_document>{self.latex_string}<end_document>

                Analyse paper by section and strictly follow this template but in JSON format:
                {{
                    {prompt_base}
                }}
                """,
            },
        ]
        return messages

    def prompt_builder(self, user_choices: list):
        """Build the prompt based on extracted information and user input."""
        self.user_choices = user_choices
        prompt_base = ""
        for count, section_heading in enumerate(self.section_headings):
            prompt_base += self._format_section_prompt(title=section_heading, num_slides=user_choices[count])

        return self.get_message_prompt(prompt_base)

    def _format_section_prompt(self, title, num_slides):
        """Format section prompts based on titles and other parameters."""
        prompt_template = f"""
            'Paper Title': 'Extracted Paper Title from LaTeX file',
            {title}: {{
                num_slides: {num_slides},
                slides_information: {{List of {num_slides} information objects, "
                    "where each object corresponds to information in that slide, "
                    "it should be in the following format:
                    [{{
                        "slide_title": "Title for each slide",
                        "slide_number": {{Slide number for this section}},
                        "speaker_notes": "250 words speech for slide",
                        "table": Extract data from latex tables to a dictionary consisting of header and rows if present in {title}, "" if absent,
                        "image": Figure path if present in {title} section (possible selections: {self.figure_list}) "" if absent,
                        "generative_prompt": Propose a cartoonist image prompt related to {title}
                    }}]
                }}
            }}
        """
        return prompt_template


if __name__ == "__main__":
    el = ExtractionLayer(file_name="examples/arXiv-2402.04616v1/complete_tex.tex")
    user_choices = [3, 1, 1, 1]
    messages = el.prompt_builder(user_choices=user_choices)
    pprint.pprint(messages)
    print(el.section_headings)
    # print(el.figure_list)
    # print(el.latex_string)
