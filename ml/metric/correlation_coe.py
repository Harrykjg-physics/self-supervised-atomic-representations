import numpy as np


def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = np.mean(x, axis=0)
    my = np.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = np.sum(xm * ym)
    x_square_sum = np.sum(xm * xm)
    y_square_sum = np.sum(ym * ym)
    r_den = np.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return np.mean(r)
