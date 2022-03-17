import { isInner } from '../Utils/isInner';

export const ODPS_CONFIGURE_PYTHON_CODE = ({
  accessId = '<Your Access Id>',
  accessSecret = '<Your Secreat Access Key>',
  project = '<ODPS Project>',
  endpoint = '<ODPS Endpoint>'
} = {}) => {
  const innerEnv = isInner();

  const akDocument = innerEnv
    ? 'Intranet environment detected, ak default set to inner d2 ak'
    : 'https://c.tb.cn/F3.ZGP28B';
  const projectDocument = innerEnv
    ? 'https://c.tb.cn/F3.ZGSHRA'
    : 'https://c.tb.cn/F3.ZGObwD';
  const endpointDocument = innerEnv
    ? 'Intranet environment detected, endpoint default set to inner odps endpoint'
    : 'endpoint document: https://c.tb.cn/F3.ZG7jad';

  return `from odps import ODPS
%load_ext odps

o = ODPS(
    access_id='${accessId}', # ak document: ${akDocument}
    secret_access_key='${accessSecret}',
    project='${project}', # project document: ${projectDocument}
    endpoint='${endpoint}' # ${endpointDocument}
)`;
};

export const ODPS_EXECUTE_PYTHON_CODE = (sql: string) => {
  return `def execute(sql):
    from tqdm.notebook import tqdm
    from time import sleep
    from IPython.display import display, HTML, clear_output
    import pandas as pd

    global _sql_execute_result, o

    if "o" not in globals():
        print("Please run odps configuration cell first")
        return

    if "_sql_execute_result" not in globals():
        _sql_execute_result = {}

    bar = tqdm(total=1, desc='Preparing sql query')
    progress = None

    instance = o.run_sql(sql)

    bar.update(1)

    display(
        HTML(
            f'<div style="font-size:13px;padding:1px">Open <a href="{instance.get_logview_address()}" target="_blank" rel="noopener norefererer">LogView</a> to checkout details</div>'
        )
    )

    finished_last_loop = 0

    while not instance.is_terminated():
        task_progress = instance.get_task_progress(instance.get_task_names())
        stages = task_progress.stages
        finished = sum(map(lambda x: x.terminated_workers, stages))
        total = sum(map(lambda x: x.total_workers, stages))
        if progress:
            if len(stages) == 0:
                progress.update(total)
            else:
                progress.update(finished - finished_last_loop)
                finished_last_loop = finished
        elif not progress and len(stages) == 0:
            continue
        else:
            progress = tqdm(total=total, desc='executing sql query')
            progress.update(finished - finished_last_loop)
            finished_last_loop = finished
        sleep(1)
    print('The data is being formatted. If the amount of data is large, it will take a while')
    df = instance.open_reader().to_pandas()

    result_key = len(_sql_execute_result.keys())
    _sql_execute_result[result_key] = df

    pd.options.display.html.table_schema = True
    pd.options.display.max_rows = None
    clear_output()
    print("you can find execute result in global variable: _sql_execute_result[{}]".format(result_key))

    return df

execute('''${sql}''')`;
};
