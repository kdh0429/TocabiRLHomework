clear all

d = load('processed_data_tocabi_mocap.txt');

start_pose = [0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, 0.0, ...
            0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0, ...
            0.0, 0.0, ...
            -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0];
        
cycle_start_idx = 31112;
cycle_end_idx = 34712;

data = zeros(cycle_end_idx-cycle_start_idx, 1 + size(start_pose,2) + 2);

data(:, 1) = (0:cycle_end_idx-cycle_start_idx-1)/2000.0;
d(:,1) = d(:,1) * (cycle_end_idx-cycle_start_idx-1)/2000.0 / d(end,1);
cur_idx = 1;
for i=1:size(data,1)
    pre_idx = cur_idx-1;
    if (pre_idx ==0)
        pre_idx=1;
    end
    for j=2:13
        data(i,j) = cubic(data(i,1), d(cur_idx,1), d(cur_idx+1,1), d(cur_idx,j+7), d(cur_idx+1,j+7), ...
                                (d(cur_idx,j+7) - d(pre_idx,j+7))/(d(cur_idx+1,1) - d(cur_idx,1)), (d(cur_idx+1,j+7) - d(cur_idx,j+7))/(d(cur_idx+1,1) - d(cur_idx,1)));
    end
    if (data(i,1) >= d(cur_idx+1,1))
        cur_idx = cur_idx + 1;
    end
end
data(:, 14:34) = repmat(start_pose(13:33), size(data,1) , 1);

d_model = load('processed_data_tocabi_walk.txt');
data(:,2) = d_model(:,2);
data(:,8) = d_model(:,8);
data(:,3) = d_model(:,3);
data(:,9) = d_model(:,9);

scale_factor = 0.5;
for i=2:13
    data(:,i) = data(:,i)*scale_factor;
end

data(:,6) = -data(:,5) - data(:,4);
data(:,7) = -data(:,3);
data(:,12) = -data(:,11) - data(:,10);
data(:,13) = -data(:,9);


%% Force Reference
f_z_max = -1025;

%DSP (left decrease, right increase)
for i=3546:3600
    data(i,35) = cubic(i-3546, 0, 320, f_z_max, 0.0,  0, 0); %f_z_max + (i-3300)/600 * abs(f_z_max);
    data(i,36) = cubic(i-3546, 0, 320, 0.0, f_z_max, 0, 0);%-(i-3300)/600 * abs(f_z_max);
end
for i=1:266
    data(i,35) = cubic(i+54, 0, 320, f_z_max, 0.0,  0, 0); %f_z_max + (i-3300)/600 * abs(f_z_max);
    data(i,36) = cubic(i+54, 0, 320, 0.0, f_z_max, 0, 0);%-(i-3300)/600 * abs(f_z_max);
end
%Right
for i=266:1746
    data(i,35) = 0.0;
    data(i,36) = f_z_max;
end
% DSP (left increase, right decrease)
for i=1746:2066
    data(i,35) = cubic(i-1746, 0, 320, 0.0, f_z_max, 0, 0);%-(i-1500)/600 * abs(f_z_max);
    data(i,36) = cubic(i-1746, 0, 320, f_z_max, 0.0, 0, 0);%f_z_max + (i-1500)/600 *abs(f_z_max);
end
% Left
for i=2066:3546
    data(i,35) = f_z_max;
    data(i,36) = 0.0;
end

save('processed_data_tocabi_mocap_with_force_reference.txt', 'data', '-ascii', '-double', '-tabs')



%%
d2_upper = d(:,21:end);

d2_mid_idx = fix(size(d,1)/2);
d_mid_idx = fix(size(data,1)/2);
for i=1:size(data,1)
    for j = 14:34
        if i < d_mid_idx
            data(i,j) = cubic(data(i,1), 0.0, data(d_mid_idx,1), d2_upper(1, j-13), d2_upper(d2_mid_idx, j-13), 0.0, 0.0);
        else
            data(i,j) = cubic(data(i,1), data(d_mid_idx,1), data(end,1), d2_upper(d2_mid_idx, j-13), d2_upper(1, j-13), 0.0, 0.0);
        end
    end
end
for i=1:size(data,1)
    for j = 27:34
        if i == fix(d_mid_idx)
            data(i,j) = -data(i+d_mid_idx, j-10);
        else
            data(i,j) = -data(fix(mod(i+d_mid_idx,size(data,1))) ,j-10);
        end
    end
end
data(:,17) = data(:,17)/2.0;
data(:,27) = data(:,27)/2.0;
data(:,18) = data(:,18)/2.0;
data(:,28) = data(:,28)/2.0;
data(:,21) = data(:,21)/1.5;
data(:,31) = data(:,31)/1.5;

save('processed_data_tocabi_mocap_with_upper.txt', 'data', '-ascii', '-double', '-tabs')