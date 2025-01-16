import re
import os
import random
import numpy as np
import pandas as pd
import joblib

import featuretools as ft
from woodwork.logical_types import Age, Categorical, Datetime

from copy import deepcopy
from pathlib import Path
from datetime import date, datetime, timedelta
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder, normalize
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from df_addons import memory_compression, df_to_excel
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

WORK_PATH = Path('Z:/python-datasets/titanic')
if not WORK_PATH.is_dir():
    WORK_PATH = Path('D:/python-datasets/titanic')

DATASET_PATH = WORK_PATH.joinpath('data')

if not WORK_PATH.is_dir():
    WORK_PATH = Path('.')
    DATASET_PATH = WORK_PATH

MODEL_PATH = WORK_PATH.joinpath('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = WORK_PATH.joinpath('predictions')
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_LOGS = WORK_PATH.joinpath('scores_local.logs')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('.')
    __file__ = Path('.')
    LOCAL_FILE = ''
else:
    LOCAL_FILE = '_local'

RANDOM_SEED = 127


def get_max_num(log_file=None):
    """Получение максимального номера итерации обучения моделей
    :param log_file: имя лог-файла с полным путем
    :return: максимальный номер
    """
    if log_file is None:
        log_file = MODELS_LOGS

    if not log_file.is_file():
        with open(log_file, mode='a') as log:
            log.write('num;mdl;fold;mdl_score;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;'
                      'f1_wght;tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        df = pd.read_csv(log_file, sep=';')
        # df.num = df.index + 1
        max_num = df.num.max()
    return max_num if max_num else 0


def clean_column_name(col_name):
    col_name = col_name.lower()  # преобразование в нижний регистр
    col_name = col_name.replace('(', '_')  # замена скобок на _
    col_name = col_name.replace(')', '')  # удаление закрывающих скобок
    col_name = col_name.replace('.', '_')  # замена точек на _
    col_name = re.sub(r'(?<=\d)_(?=\d)', '', col_name)  # удаление подчеркивания между числами
    return col_name


class DataTransform:
    def __init__(self, use_catboost=True, numeric_columns=None, category_columns=None,
                 drop_first=False, name_scaler=None, args_scaler=None):
        """
        Преобразование данных
        :param use_catboost: данные готовятся для catboost
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param drop_first: из dummy переменных удалить первую колонку
        :param name_scaler: какой скайлер будем использовать
        :param args_scaler: аргументы для скайлера, например: степень для полинома
        """
        self.use_catboost = use_catboost
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.drop_first = drop_first
        self.exclude_columns = []
        self.new_columns = []
        self.comment = {'drop_first': drop_first}
        self.transform_columns = None
        self.name_scaler = name_scaler
        self.args_scaler = args_scaler
        self.preprocess_path_file = 'reprocess_data.pkl'
        self.aggregate_path_file = 'aggregate_data.pkl'
        self.all_df = pd.DataFrame()
        self.agg_df = pd.DataFrame()
        self.scaler = None

    def cat_dummies(self, df):
        """
        Отметка категориальных колонок --> str для catboost
        OneHotEncoder для остальных
        :param df: ДФ
        :return: ДФ с фичами
        """
        # если нет цифровых колонок --> заполним их
        if self.category_columns and not self.numeric_columns:
            self.numeric_columns = [col_name for col_name in df.columns
                                    if col_name not in self.category_columns]
        # если нет категориальных колонок --> заполним их
        if self.numeric_columns and not self.category_columns:
            self.category_columns = [col_name for col_name in df.columns
                                     if col_name not in self.numeric_columns]

        for col_name in self.category_columns:
            if col_name in df.columns:
                if self.use_catboost:
                    df[col_name] = df[col_name].astype(str)
                else:
                    print(f'Трансформирую колонку: {col_name}')
                    # Create dummy variables
                    df = pd.get_dummies(df, columns=[col_name], drop_first=self.drop_first)

                    self.new_columns.extend([col for col in df.columns
                                             if col.startswith(col_name)])
        return df

    def fit_scaler(self, df):
        """
        Масштабирование цифровых колонок
        :param df: исходный ДФ
        :return: нормализованный ДФ
        """
        if not self.transform_columns:
            self.transform_columns = self.numeric_columns
        if self.name_scaler and self.transform_columns:
            print(f'Обучаю scaler: {self.name_scaler.__name__} '
                  f'с аргументами: {self.args_scaler}')
            args = self.args_scaler if self.args_scaler else tuple()
            self.scaler = self.name_scaler(*args)
            self.scaler.fit(df[self.transform_columns])

    def apply_scaler(self, df):
        """
        Масштабирование цифровых колонок
        :param df: исходный ДФ
        :return: нормализованный ДФ
        """
        if not self.transform_columns:
            self.transform_columns = self.numeric_columns
        if self.name_scaler and self.transform_columns:
            print(f'Применяю scaler: {self.name_scaler.__name__} '
                  f'с аргументами: {self.args_scaler}')
            if self.scaler is None:
                args = self.args_scaler if self.args_scaler else tuple()
                self.scaler = self.name_scaler(*args)
                scaled_data = self.scaler.fit_transform(df[self.transform_columns])
            else:
                scaled_data = self.scaler.transform(df[self.transform_columns])
            if scaled_data.shape[1] != len(self.transform_columns):
                print(f'scaler породил: {scaled_data.shape[1]} колонок')
                new_columns = [f'pnf_{n:02}' for n in range(scaled_data.shape[1])]
                df = pd.concat([df, pd.DataFrame(scaled_data, columns=new_columns)], axis=1)
                self.exclude_columns.extend(self.transform_columns)
            else:
                df[self.transform_columns] = scaled_data

            self.comment.update(scaler=self.name_scaler.__name__,
                                args_scaler=self.args_scaler)
        return df

    def fit(self, df):
        """
        Формирование фич
        :param df: исходный ФД
        :return: ДФ с агрегациями
        """
        start_time = print_msg('Группировка по целевому признаку...')

        self.fit_scaler(df)

        print_time(start_time)

        return df

    def transform(self, df, model_columns=None):
        """
        Формирование остальных фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        print(self.agg_df.info(), df.info())
        print(self.agg_df.isna().sum().sum(), df.isna().sum().sum())

        if not self.agg_df.empty:
            df = df.merge(self.agg_df, on='model', how='left')

        df = self.cat_dummies(df)

        df = self.apply_scaler(df)

        if model_columns is None:
            model_columns = df.columns.tolist()

        model_columns.extend(self.new_columns)

        exclude_columns = [col for col in self.exclude_columns if col in df.columns]
        exclude_columns.extend(col for col in df.columns if col not in model_columns)

        if exclude_columns:
            df.drop(columns=exclude_columns, inplace=True, errors='ignore')

        self.exclude_columns = exclude_columns

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df)

        return df

    def fit_transform(self, df, model_columns=None):
        """
        Fit + transform data
        :param df: исходный ФД
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        df = self.fit(df)
        df = self.transform(df, model_columns=model_columns)
        return df

    @staticmethod
    def find_title(name: str) -> str:
        """
        Функция извлечения титула из имени по правилу: пробел перед буквами и
        точка после титула
        :param name: имя пассажира
        :return: title - титул в нижнем регистре
        """
        title_search = re.search(' ([A-Za-z]+)\.', name)
        return title_search.group(1).lower() if title_search else ''

    @staticmethod
    def change_title(row: object) -> str:
        """
        Группировка титулов по категориям
        :param row: строка датафрейма
        :return: строка с новой категорией титула
        """
        if row['Sex'] == 'female':
            if row['title_cat'] in ['mme', 'countess', 'lady', 'dona', 'dr']:
                return 'mrs'
            elif row['title_cat'] in ['mlle', 'ms']:
                return 'miss'
            return row['title_cat']
        else:
            if row['title_cat'] in ['capt', 'col', 'don', 'dr', 'major', 'rev', 'sir',
                                    'jonkheer']:
                return 'other'
            return row['title_cat']

    @staticmethod
    def family_category(x: int) -> int:
        """
        Присвоение семье категории: 1-одиночка, 2-маленькая, 3-средняя, 4-большая
        :param x: количество членов семьи
        :return: категория
        """
        if x == 1:
            return 1
        elif 1 < x < 5:
            return 2
        elif 4 < x < 8:
            return 3
        else:
            return 4

    @staticmethod
    def fare_category(x: float) -> int:
        """
        Присвоение ценовой категории стоимости билета
        :param x: стоимость билета
        :return: категория билета
        """
        if x < 6.35:
            return 1
        elif 6.35 < x < 8.75:
            return 2
        elif 8.75 < x < 21.25:
            return 3
        elif 21.25 < x < 30.55:
            return 4
        elif 30.55 < x < 35.25:
            return 5
        else:
            return 6

    @staticmethod
    def age_category(x: float) -> int:
        """
        Присвоение возрастной категории пассажиру
        :param x: возраст пассажира
        :return: категория пассажира
        """
        if x < 15.8:
            return 1
        elif 15.8 < x < 24.2:
            return 2
        elif 24.2 < x < 30.8:
            return 3
        elif 30.8 < x < 36.2:
            return 4
        elif 36.2 < x < 45.2:
            return 5
        elif 45.2 < x < 63.8:
            return 6
        return 7

    # оставшихся пассажиров обработаем по общему алгоритму:
    # фильтруем по классу каюты, порту посадки, длине билета, размеру семьи и
    # получаем список пассажиров с ценой билета (fare) и
    # ценой билета / его частоту (ticket_fare) и находим билет с минимальной
    # разницей от цены билета с палубы "U".
    def get_deck(self, row: object) -> list:
        """
        Получение номера палубы в зависимости от класса, длины номера билета,
        порта посадки, количества членов семьи и пола пассажира
        :param row: строка датафрейма
        :return: список: палуба, тип сравления цены билета, разница между ценами
        """

        def filtered_df(conditions_filtered: object) -> pd.DataFrame:
            """
            Фильтрация датафрейма по передаваемым условиям
            :param conditions_filtered: условия
            :return: отфильтрованный датафрейм
            """
            df_temp = self.all_df.loc[(self.all_df['Pclass'] == row['Pclass']) &
                                      (self.all_df['deck'] != 'U') &
                                      conditions_filtered[3] &
                                      conditions_filtered[2] &
                                      conditions_filtered[1] &
                                      conditions_filtered[0]]
            return df_temp

        # если палуба известна - просто выход и возврат её
        if row['deck'] != 'U':
            return [row['deck'], 'self', 0]

        # если палуба НЕ известна - ищем по условиям
        conditions = [self.all_df['sex'] == row['sex'],
                      self.all_df['family'] == row['family'],
                      self.all_df['embarked'] == row['embarked'],
                      self.all_df['len_ticket'] == row['len_ticket'],
                      True]
        df_tmp = pd.DataFrame()
        for idx in range(len(conditions)):
            df_tmp = filtered_df(conditions)
            if len(df_tmp):
                break
            conditions[idx] = True
        df_tmp = df_tmp[['fare', 'ticket_fare', 'deck']]
        tmp_deck, name_delta, min_delta = row['deck'], 'no_found', -1
        if len(df_tmp):
            df_tmp['fare_delta'] = abs(df_tmp['fare'] - row['fare'])
            df_tmp['tf_delta'] = abs(df_tmp['ticket_fare'] - row['ticket_fare'])
            df_tmp.sort_values('fare_delta', inplace=True)
            min_f_delta = df_tmp['fare_delta'].min()
            min_t_delta = df_tmp['tf_delta'].min()
            if min_f_delta < min_t_delta:
                tmp_deck = df_tmp.loc[df_tmp['fare_delta'].idxmin(), 'deck']
                name_delta = 'f_delta'
                min_delta = min_f_delta
            else:
                tmp_deck = df_tmp.loc[df_tmp['tf_delta'].idxmin(), 'deck']
                name_delta = 't_delta'
                min_delta = min_t_delta
        return [tmp_deck, name_delta, min_delta]

    def preprocess_data(self, fill_nan=True, remake_file=False):
        """
        Предобработка данных
        :param fill_nan: заполняем пропуски в данных
        :param remake_file: формирование файлов заново / используем подготовленные ранее файлы
        :return:
        """
        preprocess_file = None

        if self.preprocess_path_file:
            preprocess_file = WORK_PATH.joinpath(self.preprocess_path_file)

        if self.preprocess_path_file and preprocess_file.is_file() and not remake_file:
            start_time = print_msg('Читаю подготовленные данные...')
            with open(preprocess_file, 'rb') as in_file:
                train, test, df_all = joblib.load(in_file)
            print_time(start_time)
            return train, test, df_all

        start_time = print_msg('Загрузка данных...')

        # объединим обучающую и тестовую выборки в один датафрейм для заполнения
        # отсутствующих значений, но перед этим добавим в тестовую недостающие поля
        train = pd.read_csv(DATASET_PATH.joinpath('train.csv'), index_col='PassengerId')
        test = pd.read_csv(DATASET_PATH.joinpath('test.csv'), index_col='PassengerId')
        train['learn'] = 1
        test['learn'] = 2
        df_all = pd.concat([train, test])

        # Исправление ошибок датасета
        # thanks to @Nadezda Demidova
        # https://www.kaggle.com/demidova/titanic-eda-tutorial-with-seaborn
        df_all.loc[df_all.index == 631, 'Age'] = 48
        # Passengers with wrong number of siblings and parch
        df_all.loc[df_all.index == 69, ['SibSp', 'Parch']] = [0, 0]
        df_all.loc[df_all.index == 1106, ['SibSp', 'Parch']] = [0, 0]

        df_all['title'] = df_all.Name.apply(self.find_title)
        df_all['len_name'] = df_all.Name.str.len()
        df_all['alone'] = df_all.apply(lambda x: 1 if x['SibSp'] + x['Parch'] == 0 else 0,
                                       axis=1)
        df_all['family'] = df_all.apply(lambda x: x['SibSp'] + x['Parch'] + 1, axis=1)
        df_all['is_cabin'] = df_all.Cabin.apply(lambda x: 'есть' if pd.notnull(x) else 'нет')
        df_all['deck'] = df_all.Cabin.fillna("U").apply(lambda x: x[0])
        df_all['deck'] = df_all.deck.apply(lambda x: "A" if x == "T" else x)

        if fill_nan:
            # Пропущены Embarked, Fare и Age
            # Embarked заполним самым часто встречающимся значением
            df_all['Embarked'].fillna(df_all['Embarked'].value_counts().index[0],
                                      inplace=True)

            # Fare заполним медианным значением для билетов этого класса каюты
            grp_fare = df_all.groupby("Pclass")["Fare"].median()
            df_all['fare'] = df_all.apply(
                lambda row: grp_fare[row["Pclass"]] if pd.isnull(row["Fare"])
                else row["Fare"], axis=1)

            # Age заполним медианным значением в зависимости от класса каюты и титула (пол
            # пассажира можно не рассматривать, т.к. он зашифрован и соответствует титулу)
            grp_age = df_all.groupby(["Pclass", "title"])["Age"].median()
            # print(grp_age)
            # у одной категории пассажиров "ms" в третьем классе отсутствует средний
            # возраст - возьмем значение из второго класса
            # print(grp_age.query('Pclass==3 and title=="ms"'))
            grp_age[3, "ms"] = grp_age[2, "ms"]
            df_all['age'] = df_all.apply(
                lambda row: grp_age[row["Pclass"], row["title"]] if pd.isnull(
                    row["Age"]) else row["Age"], axis=1)

        df_all['sex'] = df_all.Sex.apply(lambda x: 1 if x == 'female' else 0)
        idx_ticket = pd.Index(df_all["Ticket"].unique())
        df_all['ticket'] = df_all.Ticket.apply(lambda x: idx_ticket.get_loc(x))
        idx_embarked = {"C": 1, "Q": 2, "S": 3}
        df_all['embarked'] = df_all.Embarked.map(idx_embarked)
        # замена "dona" на "mrs"
        df_all.loc[df_all.title == 'dona', 'title'] = 'mrs'
        df_all['cabin'] = df_all.Cabin.apply(lambda x: 1 if pd.notnull(x) else 0)
        # # более крупная группировка по титулам
        df_all['title_cat'] = df_all.title
        df_all.title_cat = df_all.apply(self.change_title, axis=1)
        # разбивка возрастов по категориям
        df_all['idx_age'] = df_all.age.apply(self.age_category)
        # отдельно выделим пенсионеров
        df_all['retirer'] = df_all['age'].map(lambda x: 1 if x > 64 else 0)
        # разбивка семей по категориям
        df_all['idx_family'] = df_all.family.apply(self.family_category)
        #
        # длина билета
        df_all["len_ticket"] = df_all.Ticket.str.len()
        # выделение первого символа номера билета
        df_all['fst_sym_ticket'] = df_all['Ticket'].apply(lambda x: x[0])
        # Индексация билетов по первому символу номера билета
        # 5 и 8 имеют индекс выживаемости = 0
        grp = df_all[df_all.learn == 1].groupby(['fst_sym_ticket'])['Survived'].mean()
        i_ticket = grp.T.to_dict()
        df_all['idx_ticket'] = df_all.fst_sym_ticket.map(i_ticket)
        #
        # Индексация билетов по длине номера билета
        # билеты с длиной 3 и 18 имеют индекс выживаемости = 0
        grp = df_all[df_all.learn == 1].groupby(['len_ticket'])['Survived'].mean()
        i_len_ticket = grp.T.to_dict()
        df_all['idx_len_ticket'] = df_all.len_ticket.map(i_len_ticket)

        # частота использования билета: многие пассажиры путешествовали группами
        # по одному билету
        df_all['ticket_freq'] = df_all.groupby('Ticket')['Ticket'].transform('count')
        # посчитаем реальную стоимость одного билета
        df_all['ticket_fare'] = np.round(df_all.fare / df_all.ticket_freq, 1)

        # Проанализировав пассажиров (пол, состав семьи) с нулевой стоимостью билета
        # 1. можно пассажирам без кают первого класса записать палубу "A"
        df_all.loc[
            (df_all.Pclass == 1) & (df_all.Fare < 0.01) & (df_all.deck == 'U'), 'deck'] = 'A'
        # 2. можно пассажирам без кают второго класса записать палубу "D"
        df_all.loc[
            (df_all.Pclass == 2) & (df_all.Fare < 0.01) & (df_all.deck == 'U'), 'deck'] = 'D'
        # 3. можно пассажирам без кают третьего класса записать палубу "F"
        df_all.loc[
            (df_all.Pclass == 3) & (df_all.Fare < 0.01) & (df_all.deck == 'U'), 'deck'] = 'F'
        # Пассажиров 1 класса с ценой билета > 50.5 отправить на палубу "B",
        # т.к. только там были люксовые каюты
        df_all.loc[(df_all.Pclass == 1) & (df_all.ticket_fare > 50.5) &
                   (df_all.deck == 'U'), 'deck'] = 'B'
        # Пассажиров 3 класса с ценой билета < 6.2 отправить на палубу "G",
        # т.к. только там были самые дешевые билеты
        df_all.loc[(df_all.Pclass == 3) & (df_all.ticket_fare < 6.2) &
                   (df_all.deck == 'U'), 'deck'] = 'G'
        # Теперь Пассажиров 3 класса с ценой билета < 7.2 отправить на палубу "E",
        # т.к. только там были самые дешевые билеты
        df_all.loc[(df_all.Pclass == 3) & (df_all.ticket_fare < 7.2) &
                   (df_all.deck == 'U'), 'deck'] = 'E'
        # Пассажиров 2 класса с ценой билета < 6.2 отправить на палубу "E",
        # т.к. только там были самые дешевые билеты для 2 класса
        df_all.loc[(df_all.Pclass == 2) & (df_all.ticket_fare < 8.7) &
                   (df_all.deck == 'U'), 'deck'] = 'E'
        # Пассажиров 2 класса с ценой билета > 12.9 отправить на палубу "D",
        # т.к. только там были самые дорогие билеты для 2 класса
        df_all.loc[(df_all.Pclass == 2) & (df_all.ticket_fare > 13.0) &
                   (df_all.deck == 'U'), 'deck'] = 'D'
        # Пассажиров 2 класса одиноких и с портом посадки "Q" отправить на палубу "E",
        # т.к. только там были аналогичные пассажиры
        df_all.loc[(df_all.Pclass == 2) & (df_all.deck == 'U') & (df_all.Embarked == 'Q') &
                   df_all.alone, 'deck'] = 'F'

        self.all_df = df_all.copy()

        # получения номера колонки с палубой
        names_cv = df_all.columns.values.tolist()
        idx_dncv = names_cv.index('deck') + 1
        # вставка новых колонок после колонки с палубой
        df_all.insert(idx_dncv, 'delta', -1)
        df_all.insert(idx_dncv, 'name_delta', 'no_found')
        df_all.insert(idx_dncv, 'new_deck', '')
        # заполнение новых колонок для отсутствующей палубы у пассажира
        df_all.new_deck = df_all.apply(self.get_deck, axis=1)
        df_all.delta = df_all.new_deck.apply(lambda x: x[2])
        df_all.name_delta = df_all.new_deck.apply(lambda x: x[1])
        df_all.new_deck = df_all.new_deck.apply(lambda x: x[0])
        # обновление палубы
        df_all.deck = df_all.new_deck

        # после добавления 'deck' снова построим индексы палуб
        idx_deck = {d: i + 1 for i, d in enumerate(sorted(df_all.deck.unique(),
                                                          reverse=True))}
        df_all['idx_deck'] = df_all.deck.map(idx_deck)

        # пометим билеты с нулевой стоимостью
        df_all['fare_zero'] = df_all.ticket_fare.apply(lambda x: 1 if x < 0.1 else 0)
        # добавим разделение стоимости билетов по категориям
        df_all['idx_fare'] = df_all.ticket_fare.apply(self.fare_category)
        # для пробы - логарифм стоимости билета
        df_all['fare_log'] = df_all.ticket_fare.apply(lambda x: np.log(x) if x > 0 else 0)
        # добавление признака "двойное имя"
        df_all['double_name'] = df_all['Name'].str.contains('\(').astype(int)

        # разделение общего датасета на тренировочный и тестовый
        train = df_all[df_all.learn == 1].drop(columns=['learn'], axis=1)
        train.Survived = train.Survived.astype('int')
        test = df_all[df_all.learn == 2].drop(columns=['Survived', 'learn'], axis=1)

        train.reset_index(inplace=True)
        test.reset_index(inplace=True)

        # Получаем список колонок с типами 'object' и 'category'
        str_cols = test.select_dtypes(include=['object', 'category']).columns.tolist()
        # Подготовка категориальных признаков
        ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        train[str_cols] = ord_encoder.fit_transform(train[str_cols]).astype(int)
        test[str_cols] = ord_encoder.transform(test[str_cols]).astype(int)

        # df_to_excel(df_all, DATASET_PATH.joinpath('df_all.xlsx'))

        print_time(start_time)

        if self.preprocess_path_file:
            save_time = print_msg('Сохраняем предобработанные данные...')
            with open(preprocess_file, 'wb') as file:
                joblib.dump((train, test, df_all), file, compress=7)
            print_time(save_time)

        return train, test, df_all

    @staticmethod
    def drop_constant_columns(df):
        # Ищем колонки с константным значением для удаления
        col_to_drop = []
        for col in df.columns:
            if df[col].nunique() == 1:
                col_to_drop.append(col)
        if col_to_drop:
            df.drop(columns=col_to_drop, inplace=True)
        return df

    def make_agg_data(self, remake_file=False, use_featuretools=False):
        """
        Подсчет разных агрегированных статистик
        :param remake_file:
        :param use_featuretools:
        :return:
        """

        # колонки, которые будут исключены из обучения, т.к. по ним построены колонки
        # с другими признаками и категорийные колонки
        self.exclude_columns = ['learn',
                                'Name',
                                'Sex',
                                'Age',
                                'Ticket',
                                'Fare',
                                'Cabin',
                                'Embarked',
                                'new_deck',
                                'is_cabin',
                                'name_delta',
                                'deck',
                                'ticket',
                                ]

        aggregate_path_file = None

        if self.aggregate_path_file:
            aggregate_path_file = WORK_PATH.joinpath(self.aggregate_path_file)

        if self.aggregate_path_file and aggregate_path_file.is_file() and not remake_file:
            start_time = print_msg('Читаю подготовленные данные...')
            with open(aggregate_path_file, 'rb') as in_file:
                train_df, test_df = joblib.load(in_file)
            print_time(start_time)
            return train_df, test_df

        # Загрузка предобработанных данных
        train, test, df = self.preprocess_data(remake_file=remake_file)

        start_time = print_msg('Агрегация данных...')

        if use_featuretools:
            all_data = pd.concat([train[test.columns], test], ignore_index=True)

            # Создаём отношения между источниками данных
            es = ft.EntitySet(id="car_data")

            es = es.add_dataframe(dataframe_name="cars",
                                  dataframe=all_data,
                                  index="car_id",
                                  logical_types={"car_type": Categorical,
                                                 "fuel_type": Categorical,
                                                 "model": Categorical,
                                                 },
                                  )

            es = es.add_relationship("cars", "car_id", "rides", "car_id")
            es = es.add_relationship("drivers", "user_id", "rides", "user_id")
            es = es.add_relationship("cars", "car_id", "fixes", "car_id")

            # Генерируем новые признаки
            all_data, _ = ft.dfs(entityset=es,
                                 target_dataframe_name="cars",
                                 max_depth=2,
                                 )

            # Удаляем константные признаки
            all_data = ft.selection.remove_single_value_features(all_data)

            # Приведение наименований колонок в нормальный вид
            all_data.columns = all_data.columns.map(clean_column_name)

            test_df = all_data.loc[test.car_id].reset_index()
            train_df = all_data.loc[train.car_id].reset_index()
            # Добавим целевые признаки из трейна
            train_df = train_df.merge(train[['car_id', 'target_reg', 'target_class']],
                                      on=['car_id'], how='left')

        else:
            train_df, test_df = train, test

        print_time(start_time)

        if self.aggregate_path_file:
            save_time = print_msg('Сохраняем агрегированные данные...')
            with open(aggregate_path_file, 'wb') as file:
                joblib.dump((train_df, test_df), file, compress=7)
            print_time(save_time)

        return train_df, test_df


def set_all_seeds(seed=RANDOM_SEED):
    # python's seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_predict(idx_fold, model, datasets, max_num=0, submit_prefix='cb_', label_enc=None):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: roc_auc и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param label_enc: используемый label_encоder для target'а
    :return: разные roc_auc и F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, model_columns = datasets

    features2drop = ['PassengerId']

    test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

    if submit_prefix in ('tn_', 'nn_'):
        test = test.values

    print('X_train.shape', X_train.shape)
    print('train.shape', train.shape)
    print('test.shape', test.shape)

    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''

    predict_valid = model.predict(X_valid)
    predict_test = model.predict(test)

    predict_proba_classes = model.classes_

    if label_enc:
        # преобразование обратно меток классов
        # predict_valid = label_enc.inverse_transform(predict_valid.reshape(-1, 1))
        predict_test = label_enc.inverse_transform(predict_test.reshape(-1, 1))
        predict_proba_classes = label_enc.inverse_transform(
            predict_proba_classes.reshape(-1, 1)).flatten()

    try:
        valid_proba = model.predict_proba(X_valid)
        predict_proba = model.predict_proba(test)
    except:
        valid_proba = predict_valid
        predict_proba = predict_test

    # Сохранение предсказаний в файл
    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'],
                               'Survived': predict_test.flatten()})
    submission.to_csv(file_submit_csv, index=False)

    # Сохранение вероятностей в файл
    submit_proba = f'{submit_prefix}submit_proba_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_proba = PREDICTIONS_DIR.joinpath(submit_proba)
    submission_proba = pd.DataFrame(predict_proba, columns=predict_proba_classes)
    # Добавление идентификатора объекта
    submission_proba.insert(0, 'PassengerId', test_df['PassengerId'])
    submission_proba.to_csv(file_submit_proba, index=False)

    multi_class = model.get_params().get('objective', '')
    print('multi_class', multi_class)
    if multi_class == 'multiclassova':
        predict_proba = predict_proba / np.sum(predict_proba, axis=1, keepdims=True)
        valid_proba = valid_proba / np.sum(valid_proba, axis=1, keepdims=True)

    # Извлекаем вероятности для положительного класса (class 1)
    if valid_proba.shape[-1] > 1:
        valid_proba = valid_proba[:, -1]

    # Расчёт accuracy_score
    true_submit_csv = PREDICTIONS_DIR.joinpath('titanic_true_submit.csv')
    if true_submit_csv.is_file():
        true_submit = pd.read_csv(true_submit_csv)
        t_score = accuracy_score(true_submit['Survived'], submission['Survived'])
    else:
        t_score = 0

    start_item = print_msg("Расчет ROC AUC...")
    # Для многоклассового ROC AUC, нужно указать multi_class
    auc_macro = roc_auc_score(y_valid, valid_proba, average='macro')
    auc_micro = roc_auc_score(y_valid, valid_proba, average='micro')
    auc_wght = roc_auc_score(y_valid, valid_proba, average='weighted')
    print(f"auc_macro: {auc_macro}, auc_micro: {auc_micro}, auc_micro: {auc_wght}")
    print_time(start_item)

    start_item = print_msg("Расчет F1-score...")
    f1_macro = f1_micro = f1_wght = 0
    try:
        f1_macro = f1_score(y_valid, predict_valid, average='macro')
        f1_micro = f1_score(y_valid, predict_valid, average='micro')
        f1_wght = f1_score(y_valid, predict_valid, average='weighted')
    except:
        pass
    print(f'F1- f1_macro: {f1_wght:.6f}, f1_micro: {f1_wght:.6f}, f1_wght: {f1_wght:.6f}')
    print_time(start_item)

    try:
        if model.__class__.__name__ == 'CatBoostClassifier':
            eval_metric = model.get_params()['eval_metric']
            model_score = model.best_score_['validation'][eval_metric]
        else:
            model_score = accuracy_score(y_valid, predict_valid)
    except:
        model_score = 0

    return model_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, t_score


def add_info_to_log(prf, max_num, idx_fold, model, valid_scores, info_cols,
                    comment_dict=None, clf_lr=None, log_file=MODELS_LOGS):
    """
    Добавление информации об обучении модели
    :param prf: Префикс файла сабмита
    :param max_num: номер итерации обучения моделей
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param valid_scores: скоры при обучении
    :param info_cols: информативные колонки
    :param comment_dict: комментарии
    :param clf_lr: список из learning_rate моделей
    :param log_file: полный путь + имя лог файла
    :return:
    """
    m_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, score = valid_scores

    model_columns, exclude_columns, cat_columns = info_cols

    if comment_dict is None:
        comment = {}
    else:
        comment = deepcopy(comment_dict)

    feature_imp = None
    if model.__class__.__name__ == 'CatBoostClassifier':
        model_clf_lr = model.get_all_params().get('learning_rate', 0)
        feature_imp = model.feature_importances_

    elif model.__class__.__name__ == 'LGBMClassifier':
        model_clf_lr = model.get_params().get('learning_rate', 0)

    elif model.__class__.__name__ == 'XGBClassifier':
        model_clf_lr = model.get_params().get('learning_rate', 0)
    else:
        model_clf_lr = model.get_params().get('lr', None)

    if feature_imp is not None:
        use_cols = [col for col in model_columns if col not in exclude_columns]
        features = pd.DataFrame({'Feature': use_cols,
                                 'Importance': feature_imp}).sort_values('Importance',
                                                                         ascending=False)
        features.to_excel(MODEL_PATH.joinpath(f'features_{prf}{max_num}.xlsx'), index=False)

    if model_clf_lr is not None:
        model_clf_lr = round(model_clf_lr, 8)

    if clf_lr is None:
        clf_lr = model_clf_lr

    comment['clf_lr'] = clf_lr

    comment.update(model.get_params())

    prf = prf.strip('_')

    with open(log_file, mode='a') as log:
        # log.write('num;mdl;fold;mdl_score;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;'
        #           'f1_wght;tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        log.write(f'{max_num};{prf};{idx_fold};{m_score:.6f};{auc_macro:.6f};{auc_micro:.6f};'
                  f'{auc_wght:.6f};{f1_macro:.6f};{f1_micro:.6f};{f1_wght:.6f};{score:.6f};'
                  f'{model_columns};{exclude_columns};{cat_columns};{comment}\n')


def merge_submits(max_num=0, submit_prefix='cb_', num_folds=5, exclude_folds=None,
                  use_proba=False):
    """
    Объединение сабмитов
    :param max_num: номер итерации модели или список файлов, или список номеров сабмитов
    :param submit_prefix: префикс сабмита модели
    :param num_folds: количество фолдов модели для объединения
    :param exclude_folds: список списков для исключения фолдов из объединения:
                          длина списка exclude_folds должна быть равна длине списка max_num
    :param use_proba: использовать файлы с предсказаниями вероятностей
    :return: None
    """
    if use_proba:
        prob = '_proba'
    else:
        prob = ''
    # Читаем каждый файл и добавляем его содержимое в список датафреймов
    submits = pd.DataFrame()
    if isinstance(max_num, int):
        for nfld in range(1, num_folds + 1):
            submit_csv = f'{submit_prefix}submit{prob}_{max_num:03}_{nfld}{LOCAL_FILE}.csv'
            df = pd.read_csv(PREDICTIONS_DIR.joinpath(submit_csv), index_col='PassengerId')
            if use_proba:
                df.columns = [f'{col}_{nfld}' for col in df.columns]
            else:
                df.columns = [f'target_{nfld}']
            if nfld == 1:
                submits = df
            else:
                submits = submits.merge(df, on='PassengerId', suffixes=('', f'_{nfld}'))
        max_num = f'{max_num:03}'

    elif isinstance(max_num, (list, tuple)) and exclude_folds is None:
        for idx, file in enumerate(sorted(max_num)):
            df = pd.read_csv(PREDICTIONS_DIR.joinpath(file), index_col='PassengerId')
            if use_proba:
                df.columns = [f'{col}_{idx}' for col in df.columns]
            else:
                df.columns = [f'target_{idx}']
            if not idx:
                submits = df
            else:
                submits = submits.merge(df, on='PassengerId', suffixes=('', f'_{idx}'))
        max_num = '-'.join(sorted(re.findall(r'\d{3,}(?:_\d)?', ' '.join(max_num)), key=int))

    elif isinstance(max_num, (list, tuple)) and isinstance(exclude_folds, (list, tuple)):
        submits, str_nums = None, []
        for idx, (num, exc_folds) in enumerate(zip(max_num, exclude_folds), 1):
            str_num = str(num)
            for file in PREDICTIONS_DIR.glob(f'*submit{prob}_{num}_*.csv'):
                pool = re.findall(r'(?:(?<=\d{3}_)|(?<=\d{4}_))\d(?:(?=_local)|(?=\.csv))',
                                  file.name)
                if pool and int(pool[0]) not in exc_folds:
                    str_num += f'_{pool[0]}'
                    suffix = f'_{idx}_{pool[0]}'
                    df = pd.read_csv(file, index_col='car_id')
                    if use_proba:
                        df.columns = [f'{col}_{suffix}' for col in df.columns]
                    else:
                        df.columns = [f'target{suffix}']
                    if submits is None:
                        submits = df
                    else:
                        submits = submits.merge(df, on='PassengerId', suffixes=('', suffix))
            str_nums.append(str_num)
        max_num = '-'.join(sorted(str_nums))
        # print(df)
        print(max_num)

    # df.to_excel(WORK_PATH.joinpath(f'{submit_prefix}submit_{max_num}{LOCAL_FILE}.xlsx'))

    if use_proba:
        # Название классов поломок
        target_columns = sorted(set([col.rsplit('_', 1)[0] for col in submits.columns]))
        # Суммирование по классам поломок
        for col in target_columns:
            submits[col] = submits.filter(like=col, axis=1).sum(axis=1)
        # Получение имени класса поломки по максимуму из классов
        submits['Survived'] = submits.idxmax(axis=1)
    else:
        # Нахождение моды по строкам
        submits['Survived'] = submits.mode(axis=1)[0]

    submits_csv = f'{submit_prefix}submit_{max_num}{LOCAL_FILE}{prob}.csv'
    submits[['Survived']].to_csv(PREDICTIONS_DIR.joinpath(submits_csv))


if __name__ == "__main__":
    border_count = 254  # для кетбуста на ГПУ

    # Описание полей датасета:
    # PassengerId - ID пассажира
    # Survived - Пасажир выжил (1 = Да; 0 = Нет)
    # Pclass - Класс каюты пассажира (1 = Первый; 2 = Второй; 3 = Третий)
    # Name - фамилия, титул, имя
    # Sex - Пол
    # Age - Возраст
    # SibSp - Количество Братьев (Сестер) / Супругов на борту
    # Parch - Количество Родителей / Детей на борту
    # Ticket - номер билета
    # Fare - Пассажирский тариф
    # Cabin - каюта
    # Embarked - Порт посадки (C = Cherbourg; Q = Queenstown; S = Southampton)
    # # добавленные поля:
    # title - титул
    # len_name - длина имени
    # alone - Пассажир путешествовал один (True = Да; False = Нет)
    # family - количество всех членов семьи
    # is_cabin - есть каюта (есть; нет) - для визуализации
    # deck - Палуба символ
    # ticket - индекс билета (порядковый номер)
    # embarked - индекс Порт посадки
    # idx_title - индекс титула
    # cabin - есть каюта (1 = Да; 0 = Нет)
    # idx_deck - индекс Палубы
    # fare - Пассажирский тариф с заполненными пропущенными значениями
    # age - Возраст с заполненными пропущенными значениями
    # sex - Пол (1 - женский; 0 - мужской)

    # тут разные опыты с классом...

    dts = DataTransform()

    # train_data, test_data = dts.make_agg_data(remake_file=True, use_featuretools=True)
    train_data, test_data = dts.make_agg_data(remake_file=True)

    print(train_data.columns)
    cat_cols = []
    for col in train_data.columns:
        if 2 < train_data[col].nunique() < 20 and str(train_data[col].dtype)[:5] != 'float':
            cat_cols.append(col)
    print(cat_cols)

    cat_columns = ['Pclass',
                   # 'SibSp',  # Количество Братьев (Сестер) / Супругов на борту
                   # 'Parch',  # Количество Родителей / Детей на борту
                   'Embarked',
                   'title',
                   # 'family',  # Количество всех членов семьи
                   'deck',  # Палуба символ
                   'embarked',  # индекс Порт посадки
                   'title_cat',
                   'idx_age',  # разбивка возрастов по категориям
                   'idx_family',  # разбивка семей по категориям
                   # 'len_ticket',  # Длина номера билета
                   'fst_sym_ticket',
                   # 'ticket_freq',  # частота использования билета
                   'idx_deck',  # Индекс Палубы
                   'idx_fare',  # Индекс Тарифа
                   ]

    cat_columns = [col for col in cat_columns if col in test_data.columns]

    for col in cat_columns:
        print(f'{col}:', train_data[col].unique())

    # df = dts.fit(train_data)
    # print(df.columns)
    #
    # train_data = dts.fit_transform(train_data)
    # test_data = dts.transform(test_data)
    #
    # print(set(train_data.columns) - set(test_data.columns))

    # merge_submits(max_num=6, submit_prefix='cb_', use_proba=True)
