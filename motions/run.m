clc
clear all

Deg2Rad = pi/180;
Rad2Deg = 180/pi;

fname = 'humanoid3d_run.txt';
val = jsondecode(fileread(fname));
data = val.Frames;
total_time = 1.3;

for i=1:size(data,1)
    data_quat2joint(i,1) = (i-1)*3.3332000000000001e-02;
end
% root position and orientation
data_quat2joint(:,2:8) = data(:,2:8);

% right hip rotation
data_quat2joint(:,9:11) = eulerd(quaternion(data(:,17:20)),'YXZ','frame');
data_quat2joint(:,11) = -data_quat2joint(:,11);
% right knee roatation
data_quat2joint(:,12) = -data(:,21)*Rad2Deg;
% right ankle rotation
right_ankle = eulerd(quaternion(data(:,22:25)),'ZXY','frame');
data_quat2joint(:,13) = -right_ankle(:,1);
data_quat2joint(:,14) = right_ankle(:,2);

% left hip rotation
data_quat2joint(:,15:17) = eulerd(quaternion(data(:,31:34)),'YXZ','frame');
data_quat2joint(:,17) = -data_quat2joint(:,17);
% left knee roatation
data_quat2joint(:,18) = -data(:,35)*Rad2Deg;
% left ankle rotation
left_ankle = eulerd(quaternion(data(:,36:39)),'ZXY','frame');
data_quat2joint(:,19) = -left_ankle(:,1);
data_quat2joint(:,20) = left_ankle(:,2);

% chest rotation
data_quat2joint(:,21:23) = eulerd(quaternion(data(:,9:12)),'YZX','frame');
data_quat2joint(:,22) = -data_quat2joint(:,22);

% left shoulder rotation
data_quat2joint(:,24:26) = eulerd(quaternion(data(:,40:43)),'YZX','frame');
data_quat2joint(:,25) = -data_quat2joint(:,25);
data_quat2joint(:,26) = 90;
data_quat2joint(:,24) = 0.0;
% left armlink joint(Redundant)
data_quat2joint(:,27) = -90.0;
% left elbow roatation
data_quat2joint(:,28) = -data(:,44)*Rad2Deg;
% left forearm and wrist joint(Redundant)
data_quat2joint(:,29) = 0.0;
data_quat2joint(:,30) = 0.0;
data_quat2joint(:,31) = 0.0;

% neck
data_quat2joint(:,32) = 0.0;
data_quat2joint(:,33) = 0.0;

% right shoulder rotation
data_quat2joint(:,34:36) = eulerd(quaternion(data(:,26:29)),'YZX','frame');
data_quat2joint(:,35) = data_quat2joint(:,35);
data_quat2joint(:,36) = -90;
data_quat2joint(:,34)= 0.0;
% right armlink joint(Redundant)
data_quat2joint(:,37) = 90.0;
% right elbow roatation
data_quat2joint(:,38) = data(:,30)*Rad2Deg;
% right forearm and wrist joint(Redundant)
data_quat2joint(:,39) = 0.0;
data_quat2joint(:,40) = 0.0;
data_quat2joint(:,41) = 0.0;

data_quat2joint(:,9:41) = data_quat2joint(:,9:41)*Deg2Rad;


start_pose = [0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, -0.24, 0.6, -0.36, 0.0, ...
            0.0, 0.0, 0.0, ...
            0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0, ...
            0.0, 0.0, ...
            -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0];
for i=1:size(data_quat2joint,1)
    data_quat2joint(i,21:41) = start_pose(13:end);
end

save('processed_data_tocabi_run.txt', 'data_quat2joint', '-ascii', '-double', '-tabs')