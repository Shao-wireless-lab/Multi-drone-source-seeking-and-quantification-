%% Example 1
% This script runs a MOABS/DT simulation where the plume location
% is estimated via the particle filter.
% Run the simulation, collect data, plot and store the results: create object instances, run the multi-step simulation, and output the plume concentration data and the simulation animation


% -- INITIALIZE SETTINGS AND ADD FOLDER PATHS
% Initialize the environment and load the settings file example_1_settings_revised (i.e. the settings file above).
clc, clear, close all
addpath Utility
addpath Terrain
addpath Classes
example_1_settings_revised
PLOTRESULTS = 1;
TERRAIN_ON = 0;

% -- DATA STORED
fn = 'telem_testing.csv';

% -- INITIALIZE SIMULATION OBJECTS
% Create and initialize the flow field (flowfieldClass), plume (filamentClass), planner (plannerClass) and vehicle (vehicleClass) objects.
ff = flowfieldClass(ff_params);
plume = filamentClass(plume_params);
plan = plannerClass(planner_params);

% -- INITIALIZE VEHICLE / PLANNER OBJECTS
% Create multiple (10) vehicles, place their initial positions, and add the vehicles to the planner.
%TODO: how should "vehicle" be understood here? It is actually the UAVs; our simulation also has to simulate the data a UAV would collect when flying in a real scenario
% A simulation is essentially a digital emulation of a real system. Your project uses UAVs to locate the source of a methane leak, which means:
 % In reality, you would fly a UAV carrying a sensor over an area to detect the methane concentration distribution.
 % By moving and sampling with the UAV, combined with path planning and data processing, the position of the methane leak source is gradually located.
 % Therefore, in the simulation, the UAVs are the mobile carriers that actually carry out the "sampling task". Their trajectories, sampling strategies and coordination directly affect the simulation's effect and results. Even in a pure data simulation, the UAVs' behavior must be simulated to reflect the role and limitations of UAVs in real tasks.
veh = vehicleClass(vehicle_params);
swarmnum = 10; temp = ones(swarmnum,1);
tempx = linspace(-plan.w/2,plan.w/2,swarmnum)';
tempz = 2*ones(swarmnum,1);
tempy = 0*tempx;
p = plan.getPlaneGlobal([tempx,tempy,tempz],plan.r,plan.phi);
clear tempx tempy tempz
for i=1:swarmnum
    veh.p = p(i,:);
    veh = veh.setWaypoints(veh.p);
    plan = plan.addVehicle(veh);
end

% -- initialize environment
% Add the objects created above to the environment class (environmentClass) to build the overall simulation environment.
env = environmentClass();
env = env.addFlowfield(ff); clear ff;
env = env.addPlume(plume); clear plume;
env = env.addPlanner(plan); clear plan;
%if TERRAIN_ON
%    env = env.addTerrain();
%    env.terrain = env.terrain.clip2window(...
%        [env.ff.xmin,env.ff.xmax,env.ff.ymin,env.ff.ymax]);
%end
env.az = 30;
env.el = 30;

% -- main loop settings
% Enter the main simulation loop and advance through multiple time steps, performing the following
n = simparams.N*env.Fs; %10000*20, 200000 steps in total
mod_data = round(env.Fs/env.Fs_data);

dataout = [];
%fillament_locs = cell(1,n);
% Data at heights 2 and 5
%DATA_Height_1 = zeros(100*n,100*4);
%DATA_Height_2 = zeros(100*n,100*4);
%DATA_Height_3 = zeros(100*n,100*4);
%DATA_Height_4 = zeros(100*n,100*4);
DATA_Height_5 = zeros(100*n,100*4);
%DATA_Height_6 = zeros(100*n,100*4);
%DATA_Height_7 = zeros(100*n,100*4);
%DATA_Height_8 = zeros(100*n,100*4);

F1 = [];


X = [0 200 200 0];
Y = [0 0 200 200];
Z = [2,5;2,5;2,5;2,5];

% -- MAIN LOOP
%fig1 = figure(1);

% ---- Record the start time and write it to a log file (independent of the
% command window; even if the UI freezes/crashes later, this file stays on
% disk and can be inspected afterwards) ----
sim_start_time = datetime("now");
save_folder = "G=1_80_60_modified_Q_true_V7_eta_SNR_10_10000s_wind_and_quant_check"
log_filepath = fullfile(save_folder, 'timing_log.txt');
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end
fid = fopen(log_filepath, 'a');
fprintf(fid, '===== Simulation started: %s (%d steps in total) =====\n', string(sim_start_time), n);
fclose(fid);

tic
%try
for i=1:n
        % -- adjust CVT plane position  the navigation strategy is Centroidal Voronoi Tessellation
        %if i == 200
            %env.planner.rd = [40,50,5];
            %env.planner.phid = 90;
        %end

        % -- step environment
        % Update the environment state (flow field, plume diffusion, planner path, etc.).
        env = env.step();

        %DATA{i} = env.planner.dataout2;
        % Record and store the plume concentration data at sampling heights of 2m and 5m; each timestep is 100*100*4 data
        % Use the planner to collect the data at height 2m; cell2mat seems to be a function: it converts a cell array C into an ordinary matrix A
        % 1. All elements of the cell array must have the same data type and compatible dimensions. 2. During the conversion the elements of the cell array are concatenated automatically.
%        if ismember(1, env.planner.zvals)
%            DATA_Height_1(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout1)];
%        end
%        if ismember(2, env.planner.zvals)
%            DATA_Height_2(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout2)];
%        end
%        if ismember(3, env.planner.zvals)
%            DATA_Height_3(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout3)];
%        end
%        if ismember(4, env.planner.zvals)
%            DATA_Height_4(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout4)];
%        end
        if ismember(5, env.planner.zvals)
            DATA_Height_5(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout5)];
        end
%        if ismember(6, env.planner.zvals)
%            DATA_Height_6(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout6)];
%        end
%        if ismember(7, env.planner.zvals)
%            DATA_Height_7(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout7)];
%        end
%        if ismember(8, env.planner.zvals)
%            DATA_Height_8(100*(i-1) + 1:100*i,:) = [cell2mat(env.planner.dataout8)];
%        end

        %TODO: commented out for now to save memory
%        fillament_locs{i} = env.fil_locs;   % Collect the plume position data fillament_locs at every step.

        if mod(i, 300) == 0
            elapsed_sec = toc;
            avg_sec_per_step = elapsed_sec / i;
            est_remaining_sec = avg_sec_per_step * (n - i);

            fprintf('step %d/%d, elapsed=%.1f min, avg=%.4f sec/step, est_remaining=%.1f min\n', ...
                i, n, elapsed_sec/60, avg_sec_per_step, est_remaining_sec/60);

            % Also write it to the log file while at it; it does not take much I/O (written only once every 300 steps)
            fid = fopen(log_filepath, 'a');
            fprintf(fid, '%s | step %d/%d | elapsed=%.1f min | avg=%.4f sec/step | est_remaining=%.1f min\n', ...
                string(datetime("now")), i, n, elapsed_sec/60, avg_sec_per_step, est_remaining_sec/60);
            fclose(fid);
        end


end

sim_end_time = datetime("now");
total_elapsed_sec = toc;
fid = fopen(log_filepath, 'a');
fprintf(fid, '===== Main loop finished: %s, total elapsed %.2f hours (%.1f minutes) =====\n', ...
    string(sim_end_time), total_elapsed_sec/3600, total_elapsed_sec/60);
fclose(fid);
fprintf('Main loop finished, total elapsed %.2f hours\n', total_elapsed_sec/3600);