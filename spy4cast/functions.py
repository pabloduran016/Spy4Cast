import json
import os
import datetime
from typing import Optional, Dict, Any
from .stypes import Month, Slise
from time import perf_counter


__all__ = [
    'set_silence',
    'time_from_here',
    'time_to_here',
    'pretty_dict',
    'update_dataset_info_json',
    'get_dataset_info',
    'slise2str',
    'mon2str',
    'str2mon',
    'log_error',
    'debugprint',
]


# Array with the month indices
MONTH_TO_STRING = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
STRING_TO_MONTH = list(filter(lambda x: not x.startswith('_'), Month.__dict__.keys()))


class Settings:
    silence = True


def set_silence(b: bool) -> None:
    if type(b) != bool:
        raise TypeError(f'Expected bool got {type(b)}')
    Settings.silence = b


prev: Optional[float] = None
def time_from_here() -> None:
    global prev
    prev = perf_counter()
    return


def time_to_here() -> float:
    global prev
    if prev is None:
        raise ValueError('Expected to call time_from_here() before calling time_to_here()')
    rv = perf_counter() - prev
    prev = None
    return rv


def pretty_dict(d: Dict[str, str]) -> str:
    """
    Returns a string for a given dict
    :param d: dict
    :return: str
    """
    s = '('
    for k, v in d.items():
        s += f'{k}={v!r}, '
    s += ')'
    return s


def update_dataset_info_json(data_reader: type, datasets_dir: str, json_dir: str) -> None:
    data = get_dataset_info(data_reader, datasets_dir)
    with open(json_dir, 'w') as f:
        json.dump(data, f, indent=4)


def get_dataset_info(data_reader: type, d_dir: str, output_console: bool = False) -> dict[str, dict[str, str]]:
    data = {}
    for name in os.listdir(d_dir):
        if not name.endswith('.nc'):
            continue
        rd = data_reader(save_fig=False, show_plot=True,
                         dataset_dir=d_dir,
                         dataset_name=name)
        rd.load_dataset()
        info = rd.get_dataset_info()
        data[info[0]] = info[1]
        if output_console:
            print(rd.string_dataset_info)
    return data


def slise2str(slise: Slise) -> str:
    sufixes: Dict[str, str] = {'lat_min': '', 'lat_max': '', 'lon_min': '', 'lon_max': ''}
    values: Dict[str, float] = {'lat_min': slise.lat0, 'lat_max': slise.latf,
                                'lon_min': slise.lon0, 'lon_max': slise.lonf}
    for key in {'lat_min', 'lat_max'}:
        if values[key] >= 0:
            sufixes[key] = 'ºN'
        else:
            sufixes[key] = 'ºS'
        values[key] = abs(values[key])
    for key in {'lon_min', 'lon_max'}:
        if values[key] >= 0:
            sufixes[key] = 'ºE'
        else:
            sufixes[key] = 'ºW'
        values[key] = abs(values[key])
    return f'{values["lat_min"]}{sufixes["lat_min"]}, {values["lon_min"]}{sufixes["lon_min"]} - ' \
           f'{values["lat_max"]}{sufixes["lat_max"]}, {values["lon_max"]}{sufixes["lon_max"]}'


def mon2str(month: Month) -> str:
    """
    Function to return a month string depending on a month value
    :param month: Month from enum
    """
    if not 1 <= month <= 12:
        raise ValueError(f'Expected month number from 0 to 12, got {month}')
    # print('[WARNING] <month_to_string()> Changing month indexes!!!!!!!!!!!!!')
    return MONTH_TO_STRING[month - 1]


def str2mon(month: str) -> Month:
    """
    Function to return a month index depending on its string
    :param month: Value of the month. Minimum first three letters
    """
    month = month.upper()[:3]
    if not hasattr(Month, month):
        raise ValueError(f'Month not known, got {month}, valid values: {STRING_TO_MONTH}')
    return Month['JAN']


def log_error(string: str, path: Optional[str] = None) -> None:
    """
    Function to log an error in the errors foler on y_%y/m_%m/errors_%d.txt
    :param string: String to be logged
    :param path: Path to the erros folder, defult is website/log/errors
    :return:
    """
    timestamp = datetime.datetime.today()
    full_path = f'website/log/errors/{timestamp.year}/{timestamp.month}/' if path is None else path
    if not os.path.exists(full_path):
        acc = ''
        for i in full_path.split():
            acc += i
            if not os.path.exists(acc):
                os.mkdir(acc)
        assert acc == full_path, f'Expected full path to be the same as acc, got {acc=} and {full_path=}'
    print(f'[INFO] <log_error()> Logging in file with path: {os.path.join(full_path, f"errors_{timestamp.day}.txt")}')
    with open(os.path.join(full_path, f'errors_{timestamp.day}.txt'), 'a') as f:
        f.write(string + '\n')


def debugprint(*msgs: str, **kws: Any) -> None:
    if not Settings.silence:
        print(*msgs, **kws)

