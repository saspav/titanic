import re
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from time import time
from bisect import bisect_left
from print_time import print_time, print_msg

try:
    from mts_paths import WORK_PATH
except ModuleNotFoundError:
    WORK_PATH = Path('.')

__import__("warnings").filterwarnings('ignore')

file_urls = WORK_PATH.joinpath('file_urls.feather')

rus = 'АЕКМНОРСТУХВ'
eng = 'AEKMHOPCTYXB'
MAPPING = {}
for ru, en in zip(rus, eng):
    MAPPING[ru] = f'[{ru}{en}]'
    MAPPING[en] = f'[{ru}{en}]'


def mapping_symbols(text):
    return ''.join(MAPPING.get(x.upper(), x) for x in text)


def clean_column_name(col_name):
    col_name = col_name.lower()  # преобразование в нижний регистр
    col_name = col_name.replace('(', '_')  # замена скобок на _
    col_name = col_name.replace(')', '')  # удаление закрывающих скобок
    col_name = col_name.replace('.', '_')  # замена точек на _
    # name = re.sub(r"(?<=\d)_(?=\d)", "", name)  # удаление подчеркивания между числами
    return col_name


def df_to_excel(save_df, file_excel, ins_col_width=None, float_cells=None):
    """ экспорт в эксель """
    writer = pd.ExcelWriter(file_excel, engine='xlsxwriter')
    # save_df.to_excel(file_writer, sheet_name='find_RFC', index=False)
    # Convert the dataframe to an XlsxWriter Excel object.
    # Note that we turn off the default header and skip one row to allow us
    # to insert a user defined header.
    save_df.to_excel(writer, sheet_name='logs', startrow=1, header=False, index=False)
    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets['logs']
    # Add a header format.
    header_format = workbook.add_format({
        'font_name': 'Times New Roman',
        'font_size': 13,
        'bold': True,
        'text_wrap': True,
        'align': 'center',
        'valign': 'center',
        'border': 1})
    # Write the column headers with the defined format.
    worksheet.freeze_panes(1, 0)
    for col_num, value in enumerate(save_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    # вставка ссылок
    url_format = workbook.add_format({
        'font_color': 'blue',
        'underline': 1,
        'font_name': 'Times New Roman',
        'font_size': 13
    })
    cell_format = workbook.add_format()
    cell_format.set_font_name('Times New Roman')
    cell_format.set_font_size(13)
    float_format = workbook.add_format({'num_format': '0.000000'})
    float_format.set_font_name('Times New Roman')
    float_format.set_font_size(13)
    col_width = [7, 7, 10, 32, 8, 12, 12] + [32] * 9
    if ins_col_width:
        for num_pos, width in ins_col_width:
            col_width.insert(num_pos, width)
    for num, width in enumerate(col_width[:len(save_df.columns)]):
        now_cell_format = cell_format
        if float_cells and num in float_cells:
            now_cell_format = float_format
        worksheet.set_column(num, num, width, now_cell_format)
    worksheet.autofilter(0, 0, len(save_df) - 1, len(save_df.columns) - 1)
    writer.close()


def age_bucket(x):
    return bisect_left([18, 25, 35, 45, 55, 65], x)


def memory_compression(df, use_category=True, use_float=True, exclude_columns=None):
    """
    Изменение типов данных для экономии памяти
    :param df: исходный ДФ
    :param use_category: преобразовывать строки в категорию
    :param use_float: преобразовывать float в пониженную размерность
    :param exclude_columns: список колонок, которые нужно исключить из обработки
    :return: сжатый ДФ
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        # print(f'{col} тип: {tmp[col].dtype}', str(tmp[col].dtype)[:4])

        if exclude_columns and col in exclude_columns:
            continue

        if str(df[col].dtype)[:4] in 'datetime':
            continue

        elif str(df[col].dtype) not in ('object', 'category'):
            col_min = df[col].min()
            col_max = df[col].max()
            if str(df[col].dtype)[:3] == 'int':
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif col_min > np.iinfo(np.int64).min and col_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif use_float and str(df[col].dtype)[:5] == 'float':
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif (col_min > np.finfo(np.float32).min
                      and col_max < np.finfo(np.float32).max):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

        elif use_category and str(df[col].dtype) == 'object':
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Исходный размер датасета в памяти '
          f'равен {round(start_mem, 2)} мб.')
    print(f'Конечный размер датасета в памяти '
          f'равен {round(end_mem, 2)} мб.')
    print(f'Экономия памяти = {(1 - end_mem / start_mem):.1%}')
    return df


def concat_pickles(name_files, col_name=None, prefix=None):
    if name_files:
        tmp = pd.concat(pd.read_pickle(name_file) for name_file in name_files)
        # для колонки users нужно объединить списки
        if col_name and prefix:
            if 'users' in tmp.columns:
                tmp = tmp.groupby(['url_host', col_name],
                                  as_index=False).agg({f'{prefix}_count': sum,
                                                       'users': lambda x: x})
                tmp.users = tmp.users.apply(
                    lambda list2d: list2d if isinstance(list2d, set)
                    else set(x for row in list2d for x in row))
                tmp[f'{prefix}_user_count'] = tmp.users.map(len)
                tmp.drop('users', axis=1, inplace=True)
            else:
                tmp = tmp.groupby(['url_host', col_name],
                                  as_index=False)[f'{prefix}_count'].sum()
    else:
        tmp = pd.DataFrame()
    return tmp


def ratio_groups(df, col_name, prefix, codes, url_col='url_host', url_rf=True):
    # вместо получения списка url_host из большого ДФ, просто прочитаем уже
    # подготовленный ДФ с индексами url_host
    if url_rf:
        url_host = pd.read_feather(file_urls)
        if 'url_host' in url_host.columns:
            url_host.drop('url_host', axis=1, inplace=True)
        url_host.columns = [url_col]
    else:
        url_host = pd.DataFrame(df[url_col].unique(), columns=[url_col])

    prefixes = [prefix]
    if f'{prefix}_user_count' in df.columns:
        prefixes.append(f'{prefix}_user')
    for pref in prefixes:
        for cod in codes:
            tmp = df[df[col_name] == cod].rename(
                columns={f'{pref}_count': f'{pref}_{cod}_count'})
            merge_columns = [url_col, f'{pref}_{cod}_count']
            url_host = url_host.merge(tmp[merge_columns], on=url_col, how='left')
        url_host.fillna(0, inplace=True)
        total_sum = sum(url_host[f'{pref}_{cod}_count'] for cod in codes)
        for cod in codes:
            ratio_name = f'{pref}_prs_{cod}'
            count_name = f'{pref}_{cod}_count'
            url_host[ratio_name] = url_host[count_name] / total_sum
        url_host.fillna(0, inplace=True)
    # поставим тип INT для колонок с количеством
    for col in url_host.columns:
        if col.endswith('_count'):
            url_host[col] = url_host[col].astype(int)
    # посчитаем средний рейтинг между запросами и уникальными пользователями
    for cod in codes:
        ratio_cols = [f'{pref}_prs_{cod}' for pref in prefixes]
        url_host[f'{prefix}_avg_prs_{cod}'] = url_host[ratio_cols].mean(axis=1)
    return url_host


if __name__ == "__main__":
    start_time = print_msg('Объединение исходных файлов...')

    files = [WORK_PATH.joinpath(f'part_df_{i}.pkl')
             for i, _ in enumerate(glob(f'{WORK_PATH}/part*.parquet'))]
    print(files)
    all_df = concat_pickles(files)
    all_df.to_pickle(WORK_PATH.joinpath('df.pkl'))
    all_df = memory_compression(all_df)
    print_time(start_time)
