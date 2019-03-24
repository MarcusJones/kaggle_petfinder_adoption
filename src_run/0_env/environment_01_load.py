from pathlib import Path
import yaml
THIS_RUN_FOLDER = Path("~/EXPERIMENT").expanduser()

THIS_GENERATION = '0'

generation_path = THIS_RUN_FOLDER/THIS_GENERATION
assert generation_path.exists()
for pop_dir in generation_path.iterdir():
    assert pop_dir.is_dir()

    path_run_control = [f for f in pop_dir.glob("control *.json")].pop()
    with path_run_control.open() as f:
        control_dict = yaml.safe_load(f)

    pop_number = control_dict['population_number']
    individual_id = control_dict['id']

    logging.info("Gen {}, Individual {} {}".format(THIS_GENERATION, pop_number, individual_id))

discard_cols = list()
for allele in control_dict['genome']['feature set']:
    if not allele['value']:
        discard_cols.append(allele['name'])




