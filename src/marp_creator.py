"""Python file to create marp files."""

import subprocess


class MarpCreator:
    """Class to create Marp files."""

    def __init__(self) -> None:
        """Initialize the class."""
        self.slides = [
            """---
marp: true
theme: default

style: |
    .columns {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }
---"""
        ]

    def add_title_slide(self, slide_info: dict) -> None:
        """Add a title slide to the presentation."""
        self.slides.append(
            f"""
# {slide_info['title']}

----"""
        )

    def add_title_and_content_slide(self, slide_info: dict) -> None:
        """Add a title and content slide to the presentation."""
        self.slides.append(
            f"""
# {slide_info['title']}

{slide_info['content']}

----"""
        )

    def add_title_image_and_content_slide(self, slide_info: dict):
        """Add a title, image, and content slide to the presentation."""
        self.slides.append(
            f"""
# {slide_info['title']}

<div class="columns">
<div>

![width:500px]({slide_info['image']})

</div>

<div>

{slide_info['content']}
</div>
</div>

----"""
        )

    def sanitize_text(self, text: str) -> str:
        """Sanitize the text."""
        return "\n".join([x.strip() for x in text.split("\n")])

    def save_presentation(self, filename: str) -> None:
        """Save the presentation to a file."""
        self.slides.append("\n# Thank You!\n")
        with open(filename, "w") as f:
            f.write("\n".join(self.slides))

    def convert_to_images(self, filename: str, directory_name: str) -> None:
        """Convert the presentation to images."""
        subprocess.run(
            [
                "npx",
                "@marp-team/marp-cli@latest",
                "--images",
                "png",
                filename,
                "--allow-local-files",
                "--html",
                "-o",
                directory_name,
            ]
        )
