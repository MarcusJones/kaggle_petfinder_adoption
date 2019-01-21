import pandas as pd
from pathlib import Path

ref_submissions = dict()
ref_submissions['one'] = Path.cwd() / 'references' / 'Lukyanenko_Exploration_of_data_step_by_step' / 'submission.csv'
ref_submissions['two'] = Path.cwd() / 'references' / 'Scuccimarra_PetFinder Simple LGBM Baseline' / 'submission.csv'

for path_sub in ref_submissions:
    this_path = ref_submissions[path_sub]
    assert this_path.exists()

    df = pd.read_csv(this_path)


