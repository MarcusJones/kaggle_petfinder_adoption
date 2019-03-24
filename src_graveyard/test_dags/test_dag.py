from pprint import pprint
"""
{'END_DATE': '2019-01-18',
 'conf': <module 'airflow.configuration' from '/home/batman/anaconda3/lib/python3.7/site-packages/airflow/configuration.py'>,
 'dag': <DAG: python_test_parameters>,
 'dag_run': <DagRun python_test_parameters @ 2019-01-18 14:29:38.504221+00:00: manual__2019-01-18T14:29:38.504221+00:00, externally triggered: True>,
 'ds': '2019-01-18',
 'ds_nodash': '20190118',
 'end_date': '2019-01-18',
 'execution_date': <Pendulum [2019-01-18T14:29:38.504221+00:00]>,
 'inlets': [],
 'latest_date': '2019-01-18',
 'macros': <module 'airflow.macros' from '/home/batman/anaconda3/lib/python3.7/site-packages/airflow/macros/__init__.py'>,
 'next_ds': None,
 'next_execution_date': None,
 'outlets': [],
 'params': {},
 'prev_ds': None,
 'prev_execution_date': None,
 'run_id': 'manual__2019-01-18T14:29:38.504221+00:00',
 'tables': None,
 'task': <Task(PythonOperator): run_data_00>,
 'task_instance': <TaskInstance: python_test_parameters.run_data_00 2019-01-18T14:29:38.504221+00:00 [running]>,
 'task_instance_key_str': 'python_test_parameters__run_data_00__20190118',
 'templates_dict': None,
 'test_mode': False,
 'ti': <TaskInstance: python_test_parameters.run_data_00 2019-01-18T14:29:38.504221+00:00 [running]>,
 'tomorrow_ds': '2019-01-19',
 'tomorrow_ds_nodash': '20190119',
 'ts': '2019-01-18T14:29:38.504221+00:00',
 'ts_nodash': '20190118T142938.504221+0000',
 'var': {'json': None, 'value': None},
 'yesterday_ds': '2019-01-17',
 'yesterday_ds_nodash': '20190117'}
"""


def format_filename(s):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    filename = ''.join(c for c in s if c in valid_chars)
    filename = filename.replace(' ', '_')  # I don't like spaces in filenames.
    return filename

def run_data_00(**kwargs):
    print("Data 00 YES")
    pprint(kwargs['run_id'])
    run_folder = kwargs['run_id']
    run_folder = format_filename(run_folder)
    print(run_folder)



