d= load('walking_motion.txt');

start_pose = [0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, 0.0, ...
            0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0, ...
            0.0, 0.0, ...
            -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0];

cycle_start_idx = 31112;
cycle_end_idx = 34712;

data = zeros(cycle_end_idx-cycle_start_idx, 1 + size(start_pose,2) + 2);

data(1:cycle_end_idx-cycle_start_idx, 1) = (0:cycle_end_idx-cycle_start_idx-1)/2000.0;
data(1:cycle_end_idx-cycle_start_idx, 2:13) = d(cycle_start_idx:cycle_end_idx-1, 1:12);
data(1:cycle_end_idx-cycle_start_idx, 14:34) = repmat(start_pose(13:33), cycle_end_idx-cycle_start_idx , 1);

% Toe Motion
% for i=300:1500
%     data(i,6) = data(i,6) -0.174/(1500-300)*(i-300);
% end
% for i=1500:1800
%     data(i,6) = data(i,6) -0.174/(1800-1500)*(1800-i);
% end
% 
% for i=2100:3300
%     data(i,12) = data(i,12) -0.174/(3300-2100)*(i-2100);
% end
% for i=3300:3600
%     data(i,12) = data(i,12) -0.174/(3600-3300)*(3600-i);
% end

%% Force Reference
f_z_max = -1025;

%DSP (left decrease, right increase)
for i=3300:3600
    data(i,35) = cubic(i-3300, 0, 600, f_z_max, 0.0,  0, 0); %f_z_max + (i-3300)/600 * abs(f_z_max);
    data(i,36) = cubic(i-3300, 0, 600, 0.0, f_z_max, 0, 0);%-(i-3300)/600 * abs(f_z_max);
end
for i=1:300
    data(i,35) = cubic(i+300, 0, 600, f_z_max, 0.0,  0, 0); %f_z_max + (i-3300)/600 * abs(f_z_max);
    data(i,36) = cubic(i+300, 0, 600, 0.0, f_z_max, 0, 0);%-(i-3300)/600 * abs(f_z_max);
end
%Right
for i=300:1500
    data(i,35) = 0.0;
    data(i,36) = f_z_max;
end
% DSP (left increase, right decrease)
for i=1500:2100
    data(i,35) = cubic(i-1500, 0, 600, 0.0, f_z_max, 0, 0);%-(i-1500)/600 * abs(f_z_max);
    data(i,36) = cubic(i-1500, 0, 600, f_z_max, 0.0, 0, 0);%f_z_max + (i-1500)/600 *abs(f_z_max);
end
% Left
for i=2100:3300
    data(i,35) = f_z_max;
    data(i,36) = 0.0;
end
save('processed_data_tocabi_walk.txt', 'data', '-ascii', '-double', '-tabs')
%%
d2 = load('processed_data_tocabi_mocap.txt');
d2_t_scaled = d2(:,1) * (cycle_end_idx-cycle_start_idx-1)/2000.0 / d2(end,1);
d2_upper = d2(:,21:end);

d2_mid_idx = fix(size(d2,1)/2);
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

save('processed_data_tocabi_walk_with_upper.txt', 'data', '-ascii', '-double', '-tabs')