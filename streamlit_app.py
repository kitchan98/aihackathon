import os
import pickle
import tempfile
import streamlit as st
import base64
from src.utils import convert_compressed_file_to_single_latex_file
from src.extraction_layer import ExtractionLayer
from src.data_layer import DataLayer

# from src.presentation_creator import PresentationCreator
from src.marp_creator import MarpCreator
from tabulate import tabulate

import time


def convert_image(image_path):
    with open(
        image_path,
        "rb",
    ) as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def slide_row(image_data):
    for index, image_datum in enumerate(image_data):
        image_src = f"data:image/png;base64,{image_datum[2]}"
        st.image(image_src, use_column_width=True, caption=f"Slide {index + 1}")


def document_upload_app():
    """Show document upload app."""
    st.title("Document Upload App")

    uploaded_file = st.file_uploader("Upload a document", type=["zip"])

    # Simulate processing time if a file is uploaded
    if uploaded_file is not None:
        st.write("File uploaded")
        st.session_state.show_file_uploader = False

        # @TODO: Uncomment the following line to use the temporary directory
        st.session_state.temp_dir = os.path.abspath(
            tempfile.mkdtemp(prefix="temp_dir_", dir=".")
        )  # Create a temporary directory
        # st.session_state.temp_dir = "/Users/shubhamjain/personal/aihackathon/aihackathon/temp_dir_ehk1qfm5"
        # st.session_state.temp_dir = "/Users/shubhamjain/personal/aihackathon/aihackathon/temp_dir_ha9jsu5p"
        # st.session_state.temp_dir = "/Users/shubhamjain/personal/aihackathon/aihackathon/temp_dir_8a6px7mc"
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


def get_template_content(template_data) -> str:
    total_rows = len(template_data) // 2 + 1
    for row in range(total_rows):
        cols = st.columns(2)
        for col_idx, col in enumerate(cols):
            index = row * 2 + col_idx
            if index < len(template_data):
                datum = template_data[index]
                image_src = f"data:image/png;base64,{datum[2]}"
                with col:
                    st.image(image_src, width=400, caption=f"Template {index + 1}")
                    st.button(
                        f"Select Template {index + 1}",
                        key=f"select_template_{index}",
                        on_click=template_btn_clicked,
                        args=[index + 1],
                    )


def template_btn_clicked(template_index):
    """Template button clicked."""
    st.session_state.result[st.session_state.current_slide] = template_index
    if st.session_state.current_slide < len(st.session_state.result) - 1:
        st.session_state.current_slide += 1


def extract_data_from_tex():
    """Extract data from tex file."""
    messages = st.session_state.extractor.prompt_builder(st.session_state.slide_per_section)
    dl = DataLayer(save_folder="output_images")
    with st.spinner("ClaudeAI and ChatGPT are working on your request..."):
        llm_output = dl.get_completion_from_messages(messages, openAI=False, model="claude-3-sonnet-20240229")
    output_json_path = os.path.join(st.session_state.temp_dir, "output.json")
    with open(output_json_path, "w") as f:
        f.write(llm_output)
    with open(output_json_path, "r") as f:
        llm_output = f.read()
    slide_specific_data = []
    with st.spinner("Stability is in the HOUSE!! and generating images..."):
        for idx, x in enumerate(st.session_state.extractor.section_headings):
            for y in range(1, st.session_state.extractor.user_choices[idx] + 1):
                per_slide_dict = dl.llm_output_to_dict(llm_output, x, y)
                # print(per_slide_dict)
                per_slide_bullets = dl.bullet_points(per_slide_dict["speaker_notes"])
                # print(per_slide_bullets)
                per_slide_dict["image"] = dl.generate_image(
                    per_slide_dict["image"], per_slide_dict["generative_prompt"]
                )
                slide_specific_data.append([per_slide_dict, per_slide_bullets])
    slide_specific_data.insert(
        0,
        (
            {
                "slide_title": slide_specific_data[0][0]["Paper Title"],
                "image": "",
                "table": "",
                "Paper Title": slide_specific_data[0][0]["Paper Title"],
                "slide_number": 0,
            },
            [],
        ),
    )
    slide_specific_data_pkl_path = os.path.join(st.session_state.temp_dir, "slide_specific_data.pkl")
    with open(slide_specific_data_pkl_path, "wb") as f:
        pickle.dump(slide_specific_data, f)
    with open(slide_specific_data_pkl_path, "rb") as f:
        slide_specific_data = pickle.load(f)
        # st.write(slide_specific_data)
        with st.spinner("Time for one last step -- creating the presentation..."):
            create_presentation(slide_specific_data)


def create_presentation(slide_specific_data):
    """Create presentation from slide specific data."""
    st.session_state.slides_to_templates = {}
    st.session_state.slides_to_images = {}
    st.session_state.images_path = os.path.join(st.session_state.temp_dir, "images")
    os.makedirs(st.session_state.images_path, exist_ok=True)
    st.session_state.result = []
    for idx, (slide_info, slide_bullets) in enumerate(slide_specific_data):
        prs_creator = MarpCreator()
        slide_data = {
            "title": slide_info["slide_title"],
            "content": "\n".join([x for x in slide_bullets]),
            "image": (
                os.path.abspath(slide_info["image"])
                if os.path.exists(slide_info["image"])
                else os.path.join(st.session_state.extracted_folder, slide_info["image"])
            ),
            "table": (
                tabulate(slide_info["table"]["rows"], headers=slide_info["table"]["header"], tablefmt="pipe")
                if slide_info["table"] != ""
                else ""
            ),
        }
        if slide_data["content"] != "":
            prs_creator.add_title_and_content_slide(slide_info=slide_data)
            prs_creator.add_title_image_and_content_slide(slide_info=slide_data)
        else:
            prs_creator.add_title_slide(slide_info=slide_data)
        if slide_data["table"] != "":
            prs_creator.add_title_table_and_content_slide(slide_info=slide_data)
        st.session_state.slides_to_templates[idx] = prs_creator.slides
        md_path = os.path.join(st.session_state.temp_dir, f"slide_{idx}.md")
        images_path = os.path.join(st.session_state.images_path, f"slide_{idx}")
        prs_creator.save_presentation(md_path)
        prs_creator.convert_to_images(md_path, images_path)
        st.session_state.slides_to_images[idx] = sorted(
            [
                (int(x.name.split(".")[-1]), x.path, convert_image(x.path))
                for x in os.scandir(st.session_state.images_path)
                if x.name.startswith(f"slide_{idx}")
            ],
            key=lambda x: x[0],
        )[
            :-1
        ]  # remove the last image which is the last thank you slide
        st.session_state.result.append(-1)
    st.session_state.current_slide = 0


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


def previous_slide():
    """Previous slide."""
    if st.session_state.current_slide > 0:
        st.session_state.current_slide -= 1


def template_selector():
    """Template selector."""
    index = int(st.session_state.current_slide)
    st.write(f"## Select a template for the Slide #{index + 1}")
    get_template_content(st.session_state.slides_to_images[index])
    if st.session_state.current_slide > 0:
        st.button("Previous slide", on_click=previous_slide)

    # st.write(f"index = {index}")
    # st.write(f"template= {st.session_state.selected_template}")
    # st.write(st.session_state.result)
    # print(st.session_state.result)


def download_presentation():
    """Download presentation."""
    prs_creator = MarpCreator()
    for idx in range(len(st.session_state.result)):
        prs_creator.slides.append(st.session_state.slides_to_templates[idx][st.session_state.result[idx]])
    presentation_md_path = os.path.join(st.session_state.temp_dir, "marp_final.md")
    prs_creator.save_presentation(presentation_md_path)
    pdf_path = os.path.join(st.session_state.temp_dir, "marp_final.pdf")
    prs_creator.convert_to_pdf(presentation_md_path, pdf_path)
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        return pdf_bytes


def slides_app():
    """Slides app."""
    # st.write(st.session_state.result)
    cols = st.columns([0.8, 0.2])
    with cols[0]:
        st.title("Welcome to Slides Builder!")
        st.write("Select the template you want for each slide and create your draft.")
    with cols[1]:
        if st.session_state.current_slide < len(st.session_state.result):
            st.write(f"Slide {st.session_state.current_slide + 1} of {len(st.session_state.result)}")
        enable_download = len([x for x in st.session_state.result if x != -1]) == len(st.session_state.result)
        create_data = st.button(
            "Create PDF",
            disabled=(not enable_download),
        )
        if "data" not in st.session_state:
            st.session_state.data = None
        if create_data:
            # logic to create data here
            st.session_state.data = download_presentation()
            # st.write(st.session_state)
        if st.session_state.data is not None:
            st.download_button(
                "Download Presentation PDF",
                data=st.session_state.data,
                file_name="presentation.pdf",
            )
    cols = st.columns([0.3, 0.7])
    with cols[1]:
        template_selector()

    with cols[0].container(height=800):
        st.write("## Draft")
        slide_row(
            [
                st.session_state.slides_to_images[idx][selected_template - 1]
                for idx, selected_template in enumerate(st.session_state.result)
                if selected_template != -1
            ]
        )


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
        # st.write(f"Temporary directory path: {st.session_state.temp_dir}")
        page_1_placeholder.empty()
        with page_2_placeholder.container():
            outline_app()

    if st.session_state.current_page == 3:
        page_2_placeholder.empty()
        with page_3_placeholder.container():
            slides_app()


def initialize_page():
    # st.write("Initializing...")
    st.session_state.show_file_uploader = True
    st.session_state.initialized = True
    st.session_state.loading_complete = False
    st.session_state.current_slide = None
    st.session_state.prev_slide = None
    st.session_state.selected_template = None
    st.session_state.prev_selected_template = None
    st.session_state.current_page = 1
    st.session_state.total_slide_count = 0


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    if getattr(st.session_state, "initialized", None) is None:
        initialize_page()
    main()
