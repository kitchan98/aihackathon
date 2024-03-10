# slidify.ai

In this repository, we create a human-in-the-loop pipeline to generate a first draft of the presentation for any academic paper.

## Install the environment

We provide a Dockerfile to install all the relevant dependencies. To build the Docker image, run the following command:

```bash
docker build -t slidify .
```

## Run the Streamlit app

Once the image 

```bash
STABILITY_KEY=<KEY> OPENAI_API_KEY=<KEY> ANTHROPIC_API_KEY=<KEY> streamlit run streamlit_app.py
```

## Screenshots

### Upload the paper

We currently expect a zip file containing the latex files used to generate the paper. Further we expect that the primary `tex` file is named as `main.tex`.

![Upload the paper](./screenshots/upload_paper.png)