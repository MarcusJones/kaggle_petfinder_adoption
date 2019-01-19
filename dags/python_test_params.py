import airflow as af
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

# Hardcode the path to the script modules
from pathlib import Path
SCRIPT_PATH = Path("~/kaggle/petfinder_adoption/src").expanduser()
print(SCRIPT_PATH)
import sys
sys.path.insert(0, str(SCRIPT_PATH))

# Import the script modules
from test_dags.test_dag import run_data_00
from test_dags.test_dag import run_data_01


this_DAG = af.DAG('python_test_parameters',
                  schedule_interval=None,
                  start_date=datetime(2016, 1, 1))

with this_DAG as dag:
    ops = list()
    ops.append(DummyOperator(task_id='Start'))


    ops.append(PythonOperator(task_id='run_data_00',
                              python_callable=run_data_00,
                              provide_context=True))


    ops.append(PythonOperator(task_id='run_data_01',
                              python_callable=run_data_01,
                              provide_context=False,
                              ))


    ops.append(DummyOperator(task_id='End'))

# Build the DAG from the list
for i in range(len(ops)-1):
    print("{} {} >> {} {}".format(i, ops[i].task_id,i+1, ops[i+1].task_id))
    ops[i] >> ops[i+1]

