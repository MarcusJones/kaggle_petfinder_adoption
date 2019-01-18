import airflow as af
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

import sys
sys.path.insert(0, "../src")

from data.data_00 import run_data_00
from data.data_01 import run_data_01

with af.DAG('my_dag', start_date=datetime(2016, 1, 1)) as dag:
    ops = list()
    ops.append(DummyOperator(task_id='Start'))
    ops.append(DummyOperator(task_id='End'))

# Build the DAG from the list
for i in range(len(ops)-1):
    print("{} {} >> {} {}".format(i, ops[i].task_id,i+1, ops[i+1].task_id))
    ops[i] >> ops[i+1]


