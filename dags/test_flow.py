"""
Code that goes along with the Airflow tutorial located at:
https://github.com/apache/incubator-airflow/blob/master/airflow/example_dags/tutorial.py
"""
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
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
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG(
    'test_flow', default_args=default_args)

start = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag)

t2 = PythonOperator(
    task_id='sleep',
    python_callable=run_path(),
    dag=dag)

# templated_command = """
#     {% for i in range(5) %}
#         echo "{{ ds }}"
#         echo "{{ macros.ds_add(ds, 7)}}"
#         echo "{{ params.my_param }}"
#     {% endfor %}
# """
#
# t3 = BashOperator(
#     task_id='templated',
#     bash_command=templated_command,
#     params={'my_param': 'Parameter I passed in'},
#     dag=dag)

end = BashOperator(
    task_id='end',
    bash_command='echo END',
    dag=dag)

start >> t2 >> end
