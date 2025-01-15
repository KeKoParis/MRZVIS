"""
Лабораторная работа №1
Реализация модели линейной рециркуляционной 
сети с постоянным коэффициентом обучения
и ненормированными весами

Войткус С.А.
гр. 121703

Репозиторий с кодом, который был взят за основу
https://github.com/Fellooow/MRZvIS
"""
from compressor import Comp

if __name__ == '__main__':
    compressor = Comp('images_256/1.bmp', 8, 8, 20, 1500)
    compressor.process()