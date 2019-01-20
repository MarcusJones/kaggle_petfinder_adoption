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
        # print(pyfile_labels)
        if not len(pyfile_labels) >= 2:
            continue
        logging.debug("Adding {} to kernel".format(path_pyfile))
        # print(path_pyfile)
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

#%%
script_lines = list()
#%% Collect all imported tools
UTILS_DIR = Path('~/kaggle/kaggle_utils/kaggle_utils/').expanduser()
path_transformers = UTILS_DIR / 'transformers.py'
assert path_transformers.exists()
with path_transformers.open() as fh:
    utils_lines = fh.readlines()

clean_lines = list()
for line in utils_lines:
    out_line = line.replace('#%%', "#")
    out_line = out_line.replace('# %%', "#")
    clean_lines.append(out_line)

del utils_lines
clean_lines.insert(0, "#%% TRANSFORMERS\ntransfomers_module = r'''\n")
clean_lines.append("'''\n")
clean_lines.append("#%%\n")
clean_lines.append("from pathlib import Path\n")
clean_lines.append("util_outpath = Path.cwd() / 'transformers.py'\n")
clean_lines.append("with util_outpath.open('w') as fh: fh.writelines(transfomers_module)\n")
clean_lines.append("print('*** Wrote {}'.format(util_outpath))")
clean_lines.append("#%%\n")

script_lines = script_lines + clean_lines
logging.debug("Appended path_transformers {} lines".format(len(clean_lines)))

#%% Append all python files to a single script

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

#%% Convert to notebook
# Parse the script to Jupyter format
parsed = jupytext.reads("".join(script_lines), ext='.py', format_name='percent')

# Delete the file if it exists
# if out_path.exists():
#     out_path.unlink()
jupytext.writef(parsed, path_kernel_notebook_out)
logging.info("Wrote {}".format(path_kernel_notebook_out))

