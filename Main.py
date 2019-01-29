from numba import cuda

import GUI as gui

cuda.select_device(0)
dev = cuda.get_current_device()
print('CUDA device in use: ' + dev.name.decode())

gui.RunInLineWindow()

# TODO list
# Problemy:
# 1 - przy powtornym zsuwaniu obrazow program sie wykrzacza
# 2 - reczne przesuwanie obrazow nie dziala na wyswietlany obraz (buffer)