from os import listdir, rename
from os.path import isfile, join
import re
import shutil

files_dir = 'input/bleble'
files = [ '{0}/{1}'.format(files_dir, f) for f in listdir(files_dir) if isfile(join(files_dir, f)) ]

new_ser_name = 'ble'
i = 0

for f in files:
    i += 1
    f_name_match = re.search('(.+)/(.+)_(\-?[0-9]+)\.(.+)', f)
    num = f_name_match.group(3)
    if float(num) >= 0:
        minus = ''
    else:
        minus = 'minus_'
    f_ext = f_name_match.group(4)
    new_f = '{0}/{1}_{2}{3}.{4}'.format(files_dir, new_ser_name, minus, i, f_ext)
    # rename(f, new_f)
    shutil.copy(f, new_f)