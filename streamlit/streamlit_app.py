import streamlit as st
from st_click_detector import click_detector
import base64

def convert_image():
    with open("/home/ajoseaishat/Documents/streamlit-example/images/image1.png", "rb") as image_file:
        return base64.b64encode(image_file.read())

image_paths = [
    "images/image1.png",
    "images/image2.png",
    "images/image3.png",
    "images/image2.png",
    "images/image3.png",
    "images/image1.png",
    "images/image2.png",
    "images/image3.png",
    "images/image2.png",
    "images/image3.png"
]

num_slides = 5
result = [ 'Template1'] * num_slides

def slide_row(image_paths):
    h1_html = "<h1></h1>" 
    for index, image_path in enumerate(image_paths):
        h1_html += f'<span style="padding-left:20px, margin-bottom:10px"> <a href="#" id="{index}"><img  src="https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200" alt="Image" width="100"></a></span>'
    
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

    selected_image = st.empty()

    # Page 1: Document Upload
    if show_file_uploader:
        uploaded_file = page1_placeholder.file_uploader("Upload a document", type=['txt', 'pdf'])

        # Simulate processing time if a file is uploaded
        if uploaded_file is not None:
            show_file_uploader = False
            # Simulate processing time
            import time
            time.sleep(0.03)
            st.progress(100)

            st.session_state.loading_complete = True

    # Page 2: Hello World (displayed after processing is completed)
    if getattr(st.session_state, "loading_complete", False):
        # Hide Page 1 content
        page1_placeholder.empty()
        
        page2_placeholder.header("Creating your presentation ...")
        page2_placeholder.write("Choose ...")

        content = slide_row(image_paths)
        clicked = click_detector(content)
        # st.image('images/image1.png')
        button_pressed = None
        button_a = click_detector(f"<a href='#' id='Template1'><img width='20%' src='data:image/png;base64,  {convert_image()}'></a>")
        button_b = click_detector("<a href='#' id='Template2'><img width='20%' src='https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200'></a>")
        button_c = click_detector("<a href='#' id='Template3'><img width='20%' src='https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=200'></a>")

        # Container to display button content and text entry
        # container = st.empty()

        # Handle button clicks
        if button_a:
            print(clicked)
            print(int(clicked))
            result[int(clicked)] = 'Template1'
        elif button_b:
            result[int(clicked)] = 'Template1'
        elif button_c:
            result[int(clicked)] = 'Template1'
        
        # if button_pressed:
        #     container = st.container()
        #     container.header("Container Text: {}".format(button_pressed))

        # if clicked:
        #     st.image(clicked, width=100)

if __name__ == "__main__":
    main()
    print(result)
