import airflow as af
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

with af.DAG('my_dag', start_date=datetime(2016, 1, 1)) as dag:
    ops = list()
    ops.append(DummyOperator(task_id='Start'))
    ops.append(DummyOperator(task_id='End'))

# Build the DAG from the list
for i in range(len(ops)-1):
    print("{} {} >> {} {}".format(i, ops[i].task_id,i+1, ops[i+1].task_id))
    ops[i] >> ops[i+1]

print("DAG:")
dag.tree_view()