from cmath import pi

def lpf(input, prev_res, samping_freq, cutoff_freq):
    rc = 1.0 / (cutoff_freq * 2 * pi)
    dt = 1.0 / samping_freq
    a = dt / (rc + dt)
    return prev_res + a * (input - prev_res)