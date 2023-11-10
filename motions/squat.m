clear all
clc
%%
start_pose = [0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, 0.0, ...
            0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0, ...
            0.0, 0.0, ...
            -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0, ...
            0.0, 0.0, 0.92814218];
        
target_pose = [0.0, 0.0, -0.54, 1.2, -0.66, 0.0, ...
            0.0, 0.0, -0.54, 1.2, -0.66, 0.0, ...
            0.0, 0.0, 0.0, ...
            0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0, ...
            0.0, 0.0, ...
            -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0, ...
            0.0, 0.0, 0.83730532];
        
t_episode_start = 0.0;
t_squat_down_start = 1.0;
t_squat_down_end = 3.0;
t_squat_up_start = 5.0;
t_squat_up_end = 7.0;
t_episode_end = 8.0;

%%
data_hz = 40;
t = [0:1/data_hz:t_episode_end];
n_motion = size(t,2);

data = zeros(n_motion, 1 + size(target_pose,2));

% Start(steady)
for t_cur = 1:t_squat_down_start*data_hz + 1
    data(t_cur,1) = t(t_cur);
    for j = 1:size(target_pose,2)
        data(t_cur, j+1) = cubic(t(t_cur), t_episode_start, t_squat_down_start, start_pose(j), start_pose(j), 0.0, 0.0);
    end
end

% Squat Down
for t_cur = t_squat_down_start*data_hz+2:t_squat_down_end*data_hz+1
    data(t_cur,1) = t(t_cur);
    for j = 1:size(target_pose,2)
        data(t_cur, j+1) = cubic(t(t_cur), t_squat_down_start, t_squat_down_end, start_pose(j), target_pose(j), 0.0, 0.0);
    end
end

% Steady
for t_cur = t_squat_down_end*data_hz+2:t_squat_up_start*data_hz+1
    data(t_cur,1) = t(t_cur);
    for j = 1:size(target_pose,2)
        data(t_cur, j+1) = cubic(t(t_cur), t_squat_down_end, t_squat_up_start, target_pose(j), target_pose(j), 0.0, 0.0);
    end
end

% Squat Up
for t_cur = t_squat_up_start*data_hz+2:t_squat_up_end*data_hz+1
    data(t_cur,1) = t(t_cur);
    for j = 1:size(target_pose,2)
        data(t_cur, j+1) = cubic(t(t_cur), t_squat_up_start, t_squat_up_end, target_pose(j), start_pose(j), 0.0, 0.0);
    end
end

% Finish(Steady)
for t_cur = t_squat_up_end*data_hz+2:t_episode_end*data_hz+1
    data(t_cur,1) = t(t_cur);
    for j = 1:size(target_pose,2)
        data(t_cur, j+1) = cubic(t(t_cur), t_squat_up_end, t_squat_up_end, start_pose(j), start_pose(j), 0.0, 0.0);
    end
end


save('processed_data_tocabi_squat_with_pelvis.txt', 'data', '-ascii', '-double', '-tabs')