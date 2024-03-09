import streamlit as st
from st_click_detector import click_detector
import base64


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
    h1_html = "<h1>Slides</h1>"
    for index, image_path in enumerate(image_paths):
        imageBase64 = convert_image(image_path)
        image_src = f'data:image/png;base64,{imageBase64}'
        h1_html += f'<span style="padding-left:20px, margin-bottom:10px"> <a href="#" id="{index}"><img  src="{image_src}" alt="Image" width="100"></a></span>'

    h1_tag = f"<h1 style='overflow-X: auto; text-align: center; display: grid; grid-template-rows:repeat(auto-fit, minmax(200px, 1fr));grid-gap: 10px;'>{h1_html}</h1>"
    return h1_tag


def main():
    st.title("Document Upload App")

    # Placeholder for Page 1 content
    page1_placeholder = st.empty()

    # Placeholder for Page 2 content
    page2_placeholder = st.empty()

    # Boolean flag to control visibility of file uploader
    show_file_uploader = True

    # Page 1: Document Upload
    if show_file_uploader:
        uploaded_file = page1_placeholder.file_uploader(
            "Upload a document", type=["txt", "pdf"]
        )

        # Simulate processing time if a file is uploaded
        if uploaded_file is not None:
            show_file_uploader = False
            # Simulate processing time
            import time

            time.sleep(0.03)
            progress_placeholder = st.progress(100)

            st.session_state.loading_complete = True

    # Page 2: Hello World (displayed after processing is completed)
    if getattr(st.session_state, "loading_complete", False):
        
        # Hide Page 1 content
        page1_placeholder.empty()
        progress_placeholder.empty()

        page2_placeholder.header("Creating your presentation ...")
        
        content = slide_row(image_paths)
        slides_placeholder = click_detector(content)

        selected_index = 0
        if(slides_placeholder):
            selected_index = slides_placeholder
        
        template1Image = convert_image("./images/image1.png")
        template2Image = convert_image("./images/image2.png")
        template3Image = convert_image("./images/image3.png")

        template= click_detector(
            f"<a href='#' id='Template1'><img width='100' src='data:image/png;base64,{template1Image}'></a><a href='#' id='Template2'><img width='100' src='data:image/png;base64,{template2Image}'></a><a href='#' id='Template2'><img width='100' src='data:image/png;base64,{template3Image}'></a>"
        )

        # Handle button clicks
        index = int(selected_index)
        print(f'index = {index}')
        print(f'template= {template}')
        
        if template == 'Template1':
            st.session_state.result[index] = 'Template 1'
        elif template == 'Template2':
            st.session_state.result[index] = 'Template 2'
        elif template == 'Template3':
            st.session_state.result[index] = 'Template 3'

        st.write(st.session_state.result)
        print(st.session_state.result)
    
if __name__ == "__main__":
    st.session_state.result = ['Template 1', 'Template 2', 'Template 3']
    main()
