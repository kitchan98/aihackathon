import streamlit as st
from st_click_detector import click_detector
import base64

import time


def convert_image(image_path):
    with open(
        image_path,
        "rb",
    ) as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


image_paths = [
    "images/image1.png",
    "images/image2.png",
    "images/image3.png",
]

num_slides = 3
result = ["Template1"] * num_slides


def slide_row(image_paths):
    clickable_element = ""
    for index, image_path in enumerate(image_paths):
        imageBase64 = convert_image(image_path)
        image_src = f"data:image/png;base64,{imageBase64}"
        clickable_element += f'<span style="padding-left:20px, margin-bottom:10px"> <a href="#" id="{index}"><img  src="{image_src}" alt="Image" width="100"></a></span>'
    return clickable_element


def document_upload_app(placeholder):
    """Show document upload app."""
    placeholder.title("Document Upload App")

    uploaded_file = placeholder.file_uploader("Upload a document", type=["txt", "pdf"])

    # Simulate processing time if a file is uploaded
    if uploaded_file is not None:
        placeholder.write("File uploaded")
        st.session_state.show_file_uploader = False

        # Simulate processing time
        time.sleep(0.03)
        # progress_placeholder = st.progress(100)
        st.session_state.loading_complete = True


def slides_app(placeholder):
    placeholder.write("# Selected slides")
    content = slide_row(image_paths)
    slides_placeholder = click_detector(content, key="selected_slide")

    if slides_placeholder:
        st.session_state.selected_index = slides_placeholder
        st.session_state.selected_template = None

    if st.session_state.selected_index is not None:
        template1Image = convert_image("./images/image1.png")
        template2Image = convert_image("./images/image2.png")
        template3Image = convert_image("./images/image3.png")

        st.session_state.selected_template = click_detector(
            f"<a href='#' id='Template1'><img width='100' src='data:image/png;base64,{template1Image}'></a>"
            f"<a href='#' id='Template2'><img width='100' src='data:image/png;base64,{template2Image}'></a>"
            f"<a href='#' id='Template3'><img width='100' src='data:image/png;base64,{template3Image}'></a>",
            key="selected_template_index",
        )

        # Handle button clicks
        index = int(st.session_state.selected_index)
        st.write(f"index = {index}")
        st.write(f"template= {st.session_state.selected_template}")
        st.write("-------")

        if st.session_state.selected_template is not None:
            template = st.session_state.selected_template
            if template == "Template1":
                st.session_state.result[index] = "Template 1"
            elif template == "Template2":
                st.session_state.result[index] = "Template 2"
            elif template == "Template3":
                st.session_state.result[index] = "Template 3"

        st.write(st.session_state.result)
        print(st.session_state.result)


def main():
    # Page 1: Document Upload
    page_1_placeholder = st.empty()
    page_2_placeholder = st.empty()
    if st.session_state.show_file_uploader:
        document_upload_app(page_1_placeholder)
    # Page 2: Hello World (displayed after processing is completed)
    if st.session_state.loading_complete:
        page_1_placeholder.empty()
        print(st.session_state)
        page_2_placeholder.header("Creating your presentation ...")
        slides_app(page_2_placeholder)


def initialize_page():
    st.write("Initializing...")
    st.session_state.show_file_uploader = True
    st.session_state.initialized = True
    st.session_state.loading_complete = False
    st.session_state.selected_index = None
    st.session_state.selected_template = None
    st.session_state.result = ["Template 1", "Template 2", "Template 3"]


if __name__ == "__main__":
    if getattr(st.session_state, "initialized", None) is None:
        initialize_page()
    main()
