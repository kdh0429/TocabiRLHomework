def cubic(time,time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):
    x_t = x_0
    if (time < time_0):
        x_t = x_0
    elif (time > time_f):
        x_t = x_f
    else :
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time2 * total_time
        total_x    = x_f - x_0

        x_t = x_0 + x_dot_0 * elapsed_time \
            + (3 * total_x / total_time2 \
            - 2 * x_dot_0 / total_time \
            - x_dot_f / total_time) \
            * elapsed_time * elapsed_time \
            + (-2 * total_x / total_time3 + \
            (x_dot_0 + x_dot_f) / total_time2) \
            * elapsed_time * elapsed_time * elapsed_time
    return x_t

def cubicDot(time, time_0, time_f, x_0, x_f, x_dot_0, x_dot_f):

    if (time < time_0):
        x_t = x_dot_0
    elif (time > time_f):
        x_t = x_dot_f
    else:
        elapsed_time = time - time_0
        total_time = time_f - time_0
        total_time2 = total_time * total_time
        total_time3 = total_time2 * total_time
        total_x    = x_f - x_0

        x_t = x_dot_0 \
            + 2*(3 * total_x / total_time2 \
            - 2 * x_dot_0 / total_time \
            - x_dot_f / total_time) \
            * elapsed_time \
            + 3*(-2 * total_x / total_time3 +  \
            (x_dot_0 + x_dot_f) / total_time2) \
            * elapsed_time * elapsed_time

    return x_t
