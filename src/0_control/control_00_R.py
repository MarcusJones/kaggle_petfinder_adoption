from pathlib import  Path
import yaml
THIS_RUN_FOLDER = "~/EXPERIMENT"
THIS_GENERATION = '0'
THIS_POP_NUMBER = '000'
THIS_IND = '3164401158493596485'
THIS_FILE_NAME = "control {}.json".format(THIS_IND)
path_run = Path(THIS_RUN_FOLDER).expanduser() / THIS_GENERATION / "{}_{}".format(THIS_POP_NUMBER, THIS_IND)
assert path_run.exists()
path_run_file = path_run / THIS_FILE_NAME
assert path_run_file.exists()
with path_run_file.open() as f:
    control_dict = yaml.safe_load(f)
