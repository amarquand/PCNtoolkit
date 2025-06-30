import glob
import os
import shutil
import subprocess

EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'examples'))
TUTORIALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pages', 'tutorials'))

os.makedirs(TUTORIALS_DIR, exist_ok=True)

# Clean up old rst and support files in tutorials dir
def clean_tutorials_dir():
    for f in glob.glob(os.path.join(TUTORIALS_DIR, '*.rst')):
        os.remove(f)
    for d in glob.glob(os.path.join(TUTORIALS_DIR, '*_files')):
        shutil.rmtree(d)

def convert_notebooks():
    notebooks = glob.glob(os.path.join(EXAMPLES_DIR, '*.ipynb'))
    for nb_path in notebooks:
        nb_name = os.path.splitext(os.path.basename(nb_path))[0]
        rst_path = os.path.join(TUTORIALS_DIR, f'{nb_name}.rst')
        # Convert notebook to rst
        subprocess.run([
            'jupyter', 'nbconvert', '--to', 'rst', nb_path,
            '--output', nb_name, '--output-dir', TUTORIALS_DIR
        ], check=True)

if __name__ == '__main__':
    clean_tutorials_dir()
    convert_notebooks() 