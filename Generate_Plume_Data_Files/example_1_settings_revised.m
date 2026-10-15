% -- set simulation params
% Parameter configuration: define the flow field, plume, path planning, vehicle, simulation duration, etc.
% 10000 * 20 = 200000 steps
simparams.N = 10000;  % Number of simulation time steps / parameter related to the simulation duration.
simparams.modnum = 20;  %sampling frequency

% -- create param structs for initialization
ff_params               = struct;
ff_params.dim           = 3;    % indicates a 3D flow field
ff_params.uv0           = [1,0];    % mean advection flow vw=[1.0], the mean part of the advection model
% ff_params.abG           = [0.005,0.02,1];   % turbulence, diffusion coefficients, or related flow-field characteristic parameters
ff_params.abG           = [0, 0, 0];  % configuration with no disturbance
% ff_params.abG           = [0,0,0];
%ff_params.uv0           = [1,-1];
%ff_params.abG           = [0.5,0.1,3];
%ff_params.uv0           = [1,0.5];
%ff_params.abG           = [0.4,0.4,5];
%ff_params.alpha          = 0.3;

ff_params.Fs = 20;  % sensor data frequency

plume_params            = struct;
 plume_params(1,1).p0    = [80 60 0];   % set initial position 3 dim
% plume_params(1,1).p0    = [60 120 0];   % set initial position 3 dim
%plume_params(1,1).p0    = [30 150 0];   % set initial position 3 dim
plume_params(1,1).Collisions = 'False';
plume_params.Fs = 20;

planner_params          = struct;
planner_params.NavStrategy = 'GridSampling'; % The navigation strategy is normally Centroidal Voronoi Tessellation; for data generation it must be changed to GridSampling
planner_params.QuantStrategy = 'Mod-NGI';   % quantification strategy, possibly some kind of grid or sampling strategy
planner_params.r        = [40,50,2];    % r, rd: size of the planning region or sensing radius
planner_params.rd       = [40,50,2];
planner_params.w        = 30;   % w, h: width and height of the region
planner_params.h        = 10;
planner_params.phi      = 90;   % angle-related parameter, possibly heading or field-of-view angle
planner_params.phid     = 90;

planner_params.delx = 2;    % spatial discretization step
planner_params.dely = 2;
%planner_params.delz = 2/1.5;

% Define the spatial sampling point positions; only two heights, 2 and 5, are chosen on the z axis
%planner_params.dataout2 = zeros(200/planner_params.delx,200/planner_params.dely,20/planner_params.delz,simparams.N * ff_params.Fs,2);
%planner_params.dataout2 = zeros(200/planner_params.delx,200/planner_params.dely,20/planner_params.delz,simparams.N * ff_params.Fs,2);
planner_params.xvals = planner_params.delx/2:planner_params.delx:200;   % start:step:end
planner_params.yvals = planner_params.dely/2:planner_params.dely:200;
%planner_params.zvals = planner_params.delz/2:planner_params.delz:20;
planner_params.zvals = [5];  %[2,5]; the code indexes by index, so if I want 3 and 4, kk should be passed as [3,4]

vehicle_params          = struct;
vehicle_params.QuantifyFlag = 1;    % indicates that the quantification function or some sensing function is enabled