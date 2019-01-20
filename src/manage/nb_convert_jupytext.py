"""
Use this script to convert all *.py files to jupyter notebook format
"""

#%%
from pathlib import Path
import jupytext
import logging
logging.getLogger().setLevel(logging.DEBUG)
import shutil
from pprint import pprint
#%%
# Get the IPython script root dir
root_dir = Path.cwd()
path_ipy_root=root_dir / "src"
path_kernel_script_out=root_dir / "kernel_submission" / "kernel.py"
path_kernel_notebook_out=root_dir / "kernel_submission" / "kernel.ipynb"

#%% Get the folders
script_folders = list()
for folder in [f for f in path_ipy_root.glob('./*/') if f.is_dir() == True]:
    folder_number = folder.parts[-1].split('_')[0]
    if not folder_number.isdigit():
        continue
    print(folder)
    script_folders.append(folder)
script_folders.sort()

#%% Iterate over files inside the folders
kernel_files = dict()
for folder in script_folders:
    logging.debug("Script folder {}".format(folder))
    kernel_files[folder.stem] = list()
    for path_pyfile in folder.glob("*.py"):

        pyfile = path_pyfile.stem

        if pyfile == '__init__':
            continue

        pyfile_labels = pyfile.split('_')
        print(pyfile_labels)
        if not len(pyfile_labels) >= 2:
            continue
        logging.debug("Adding {} to kernel".format(path_pyfile))
        print(path_pyfile)
        kernel_files[folder.stem].append(path_pyfile)

#%% Sort the result
for k in kernel_files:
    kernel_files[k].sort()

#%% Print
print("*** Kernel manifest ***")
for k in kernel_files:
    print(k)
    for f in kernel_files[k]:
        print('\t', f.stem)

#%% Append all python files to a single script
script_lines = list()
for k in kernel_files:
    for f in kernel_files[k]:
        with f.open() as fh:
            lines = fh.readlines()
        # Concatenate
        script_lines = script_lines + lines
        logging.debug("Added {:5} lines from {}".format(len(lines),f.stem))
#%%
with path_kernel_script_out.open('w') as fh:
    fh.writelines(script_lines)
logging.debug("Wrote {}".format(path_kernel_script_out))
#%%

processed_cnt = 0
# Walk the directory tree
for item in path_ipy_root.iterdir():
    if not item.is_dir():
        continue
    if not item.parts[-1] in notebook_folders:
        continue

    for script_path in item.glob('*.py'):
        logging.info("Processing: ./{}/{}".format(script_path.parts[-2], script_path.parts[-1]))

        # Select (make) the output folder
        jupyter_dir_name = script_path.parts[-2]
        jupyter_path = path_jupyter_root / jupyter_dir_name
        jupyter_path.mkdir(parents=True, exist_ok=True)

        # The output path
        jupyter_file_name = script_path.stem + '.ipynb'
        out_path = jupyter_path / jupyter_file_name

        script_lines = script_path.read_text()

        # Concatenate the header
        total_script_lines = header_lines + script_lines
        logging.info("{}+{}={} total lines".format( len(header_lines.split('\n')), len(script_lines.split('\n')), len(total_script_lines.split('\n')), ))

        # Parse the script to Jupyter format
        parsed = jupytext.reads(total_script_lines, ext='.py', format_name='percent')

        # Delete the file if it exists
        if out_path.exists():
            out_path.unlink()
        logging.info("Wrote {}".format(out_path))
        jupytext.writef(parsed, out_path)
        processed_cnt +=1


logging.info("Wrote {} files".format(processed_cnt))
