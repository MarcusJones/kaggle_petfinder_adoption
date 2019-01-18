import airflow as af
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime

with af.DAG('my_dag', start_date=datetime(2016, 1, 1)) as dag:
    op = DummyOperator(task_id='op')

op.dag is dag # True

