"""
Code that goes along with the Airflow tutorial located at:
https://github.com/apache/incubator-airflow/blob/master/airflow/example_dags/tutorial.py
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import airflow
print(airflow.__version__)
from datetime import datetime, timedelta
import os
from runpy import run_path

def conditional(cond, warning=None):
    def noop_decorator(func):
        return func  # pass through

    def neutered_function(func):
        def neutered(*args, **kw):
            if warning:
                log.warn(warning)
            return
        return neutered

    return noop_decorator if cond else neutered_function

run_settings = {
    'data1':1,
    'data2':1,
    'features1':1,
    'models1':1,
    'visualization1':1,
}

for this_set in run_settings:
    os.environ[this_set] = str(run_settings[this_set])


def conditional(cond, warning=None):
    def noop_decorator(func):
        return func  # pass through

    def neutered_function(func):
        def neutered(*args, **kw):
            if warning:
                log.warn(warning)
            return
        return neutered

    return noop_decorator if cond else neutered_function


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'end_data': None,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

def my_test():
    print("Test print")


with DAG('my_dag', start_date=datetime(2016, 1, 1)) as dag:
    op = DummyOperator('op')

op.dag is dag # True


# dag = DAG('test_flow',
#           catchup=False,
#           schedule_interval=None,
#           default_args=default_args
#           )
# with DAG('test_flow', catchup=False, schedule_interval=None, default_args=default_args ) as dag:
#     # start = DummyOperator('op')
#     op = DummyOperator('op')

#     t0 = BashOperator(task_id='echo', bash_command='echo Starting this DAG')
#
#     t1 = BashOperator(task_id='cd', bash_command='cd ~/kaggle/petfinder_adoption', dag=dag)
#
#     t2 = PythonOperator(task_id='call_python', python_callable=my_test)
#
#     end = BashOperator(task_id='end', bash_command='echo END')
#
# start >> t0 >> t1 >> t2 >> end
# start >> end
