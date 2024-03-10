import os
import pickle
import tempfile
from typing import Dict
import streamlit as st
from st_click_detector import click_detector
import base64
from src.utils import convert_compressed_file_to_single_latex_file
from src.extraction_layer import ExtractionLayer
from src.data_layer import DataLayer
from src.presentation_creator import PresentationCreator
from src.marp_creator import MarpCreator
from tabulate import tabulate

import time


def convert_image(image_path):
    with open(
        image_path,
        "rb",
    ) as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


num_slides = 3
result = ["Template1"] * num_slides

image_name_to_path = {"image_1": "images/image1.png", "image_2": "images/image2.png", "image_3": "images/image3.png"}


def slide_row(image_paths):
    clickable_element = ""
    for index, image_path in enumerate(image_paths):
        imageBase64 = convert_image(image_path)
        image_src = f"data:image/png;base64,{imageBase64}"
        clickable_element += (
            f'<a href="#" id="{index}" >'
            f'<img  src="{image_src}" alt="Image" width="120px" style="padding-right:10px"></a>'
        )
    return clickable_element


def document_upload_app():
    """Show document upload app."""
    st.title("Document Upload App")

    uploaded_file = st.file_uploader("Upload a document", type=["zip"])

    # Simulate processing time if a file is uploaded
    if uploaded_file is not None:
        st.write("File uploaded")
        st.session_state.show_file_uploader = False

        st.session_state.temp_dir = tempfile.mkdtemp()
        bytes_data = uploaded_file.getvalue()
        uploaded_file_path = os.path.join(st.session_state.temp_dir, "uploaded_file.zip")
        with open(uploaded_file_path, "wb") as f:
            f.write(bytes_data)
        st.session_state.complete_latex = os.path.join(st.session_state.temp_dir, "complete_tex.tex")
        st.session_state.extracted_folder = convert_compressed_file_to_single_latex_file(
            uploaded_file_path, st.session_state.temp_dir, st.session_state.complete_latex
        )

        # Simulate processing time
        time.sleep(0.03)
        # progress_placeholder = st.progress(100)
        st.session_state.current_page = 2


def get_template_content(template_data: Dict) -> str:
    element = ""
    for key, val in template_data.items():
        element += f"<a href='#' id='{key}'><img width='100' src='data:image/png;base64,{convert_image(val)}'></a>"
    return element


def extract_data_from_tex():
    """Extract data from tex file."""
    messages = st.session_state.extractor.prompt_builder(st.session_state.slide_per_section)
    dl = DataLayer(save_folder="output_images")
    # llm_output = dl.get_completion_from_messages(messages, openAI=False, model="claude-3-sonnet-20240229")
    # with open("output.json", "w") as f:
    # f.write(llm_output)
    with open("output.json", "r") as f:
        llm_output = f.read()
    # slide_specific_data = []
    # for idx, x in enumerate(st.session_state.extractor.section_headings):
    #     for y in range(1, st.session_state.extractor.user_choices[idx] + 1):
    #         per_slide_dict = dl.llm_output_to_dict(llm_output, x, y)
    #         # print(per_slide_dict)
    #         per_slide_bullets = dl.bullet_points(per_slide_dict["speaker_notes"])
    #         # print(per_slide_bullets)
    # per_slide_dict["image"] = dl.generate_image(per_slide_dict["image"], per_slide_dict["generative_prompt"])
    #         slide_specific_data.append([per_slide_dict, per_slide_bullets])
    # with open("slide_specific_data.json", "wb") as f:
    #     pickle.dump(slide_specific_data, f)
    with open("slide_specific_data.json", "rb") as f:
        slide_specific_data = pickle.load(f)
        st.write(slide_specific_data)
        create_presentation(slide_specific_data)


def create_presentation(slide_specific_data):
    """Create presentation from slide specific data."""
    # prs_creator = PresentationCreator()
    prs_creator = MarpCreator()
    for slide_info, slide_bullets in slide_specific_data:
        slide_data = {
            "title": slide_info["slide_title"],
            "content": "\n".join([x for x in slide_bullets]),
            "image": (
                slide_info["image"]
                if os.path.exists(slide_info["image"])
                else os.path.join(st.session_state.extracted_folder, slide_info["image"])
            ),
            "table": tabulate(slide_info["table"]["rows"], headers=slide_info["table"]["header"], tablefmt="pipe")
        }
        # prs_creator.add_title_content_layout(
        #     slide_info=slide_data,
        # )
        # prs_creator.add_picture_caption_layout(slide_info=slide_data)
        prs_creator.add_title_and_content_slide(slide_info=slide_data)
        prs_creator.add_title_image_and_content_slide(slide_info=slide_data)
    # prs_creator.save_presentation("presentation_sample.pptx")
    prs_creator.save_presentation("presentation_marp.md")
    # prs_creator.convert_presentation_to_image("presentation_sample.pptx", "presentation_images")


def outline_app():
    st.title("Outline")

    st.session_state.section_titles = st.session_state.extractor.section_headings

    section_titles = st.session_state.section_titles
    num_columns = 4
    for i in range(len(section_titles) // num_columns + 1):
        cols = st.columns(4)
        for columns in range(4):
            if i * 4 + columns < len(section_titles):
                section = section_titles[i * 4 + columns]
                cols[columns].selectbox(
                    section_titles[i * 4 + columns], ("1", "2", "3", "4"), key=f"num_slides_{section}"
                )

    st.button("Load Slides", on_click=outline_btn_clicked)


def outline_btn_clicked():
    st.session_state.slide_per_section = [
        int(st.session_state[f"num_slides_{section}"]) for section in st.session_state.section_titles
    ]
    extract_data_from_tex()
    st.session_state.current_page = 3


def slides_app():
    st.header("Let's create your presentation!")
    st.write("# Selected slides")

    slides_placeholder = st.empty()
    with slides_placeholder:
        content = slide_row([image_name_to_path[x] for x in st.session_state.result])
        click_detector(content, key="selected_slide")

    if st.session_state.selected_slide is not None:
        if st.session_state.selected_slide != st.session_state.prev_selected_slide:
            st.session_state.selected_template = None
        st.session_state.prev_selected_slide = st.session_state.selected_slide
        # Handle button clicks
        index = int(st.session_state.selected_slide)
        st.write(f"# Select a template for the Slide #{index + 1}")
        click_detector(get_template_content(image_name_to_path), key="selected_template")

        st.write(f"index = {index}")
        st.write(f"template= {st.session_state.selected_template}")

        if st.session_state.selected_template is not None:
            if st.session_state.selected_template != st.session_state.prev_selected_template:
                st.session_state.prev_selected_template = st.session_state.selected_template
                st.session_state.result[index] = st.session_state.selected_template
                # slides_placeholder.empty()
                st.experimental_rerun()

        st.write(st.session_state.result)
        print(st.session_state.result)


def setup_extractor():
    """Setup the extractor."""
    st.session_state.extractor = ExtractionLayer(file_name=st.session_state.complete_latex)


def main():
    # Page 1: Document Upload
    page_1_placeholder = st.empty()
    page_2_placeholder = st.empty()
    page_3_placeholder = st.empty()

    if st.session_state.current_page == 1:
        with page_1_placeholder.container():
            document_upload_app()

    if st.session_state.current_page == 2:
        setup_extractor()
        st.write(f"Temporary directory path: {st.session_state.temp_dir}")
        page_1_placeholder.empty()
        with page_2_placeholder.container():
            outline_app()

    if st.session_state.current_page == 3:
        page_2_placeholder.empty()
        with page_3_placeholder.container():
            slides_app()


def initialize_page():
    st.write("Initializing...")
    st.session_state.show_file_uploader = True
    st.session_state.initialized = True
    st.session_state.loading_complete = False
    st.session_state.selected_slide = None
    st.session_state.prev_selected_slide = None
    st.session_state.selected_template = None
    st.session_state.prev_selected_template = None
    st.session_state.result = list(image_name_to_path.keys())
    st.session_state.current_page = 1
    st.session_state.selected_index = None
    st.session_state.total_slide_count = 0


if __name__ == "__main__":
    if getattr(st.session_state, "initialized", None) is None:
        initialize_page()
    main()
