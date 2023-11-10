
function y = cubic(time,time_0, time_f, x_0, x_f, x_dot_0, x_dot_f)
    if (time < time_0)
        y = x_0;
    elseif (time > time_f)
        y = x_f;
    else
        elapsed_time = time - time_0;
        total_time = time_f - time_0;
        total_time2 = total_time * total_time;
        total_time3 = total_time2 * total_time;
        total_x    = x_f - x_0;

        y = x_0 + x_dot_0 * elapsed_time + (3 * total_x / total_time2 - 2 * x_dot_0 / total_time ...
            - x_dot_f / total_time) * elapsed_time * elapsed_time ...
            + (-2 * total_x / total_time3 + (x_dot_0 + x_dot_f) / total_time2) * elapsed_time * elapsed_time * elapsed_time;
    end
end