from time import time


def convert_seconds(time_apply, time_format=False):
    # print(type(time_apply), time_apply)
    try:
        time_apply = float(time_apply)
    except ValueError:
        time_apply = 0
    if isinstance(time_apply, (int, float)):
        hrs = time_apply // 3600
        mns = time_apply % 3600
        sec = mns % 60
        time_string = ''
        if hrs:
            time_string = f'{hrs:.0f} час '
        if mns // 60 or hrs:
            time_string += f'{mns // 60:.0f} мин '
        if time_format:
            return f'{int(hrs)}:{int(mns // 60):02}:{int(sec):02}'
        return f'{time_string}{sec:.1f} сек'


def print_time(time_start, title=''):
    """
    Печать времени выполнения процесса
    :param time_start: время запуска в формате time.time()
    :param title: заголовок для сообщения
    :return:
    """
    title = f'{title} --> ' if title else ''
    time_apply = time() - time_start
    print(f'{title} Время обработки: {convert_seconds(time_apply)}'.strip())
    return convert_seconds(time_apply, time_format=True)


def print_msg(msg):
    print(msg)
    return time()


if __name__ == "__main__":
    print(convert_seconds('8.6', time_format=True))
