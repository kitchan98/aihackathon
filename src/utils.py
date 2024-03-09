"""Utility functions for the project."""

import os
import shutil
import subprocess


def convert_compressed_file_to_single_latex_file(input_file_path, output_dir, output_file_path):
    """Convert tar.gz file to single LaTeX file."""
    shutil.unpack_archive(input_file_path, output_dir)
    extracted_folder = [x.path for x in os.scandir(output_dir) if (x.is_dir() and (not x.name.startswith("__")))][0]
    latexpand_path = os.path.join(os.path.dirname(__file__), "../", "latexpand")
    process = subprocess.run(["perl", latexpand_path, "main.tex", "-o", output_file_path], cwd=extracted_folder)
    if process.check_returncode() == 0:
        raise Exception("Error in converting compressed file to single LaTeX file.")
