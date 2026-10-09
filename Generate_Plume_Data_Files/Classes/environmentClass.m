%% Environment Class
% ----------------------
% by Derek Hollenbeck
%
% This class creates an environment object that acts as a container for
% running the MOABS/DT simulation
%
% EXAMPLE:
%   ...
% ----------------------
%This class is an abstraction of the "environment", integrating the flow field that simulates gas (e.g. methane) diffusion, the pollutant plume, the UAV path planner, and the terrain information.
 %It manages the creation of these submodules, their state updates (step), execution of the simulation loop, and visualization of the results.
 %It lets callers operate the complex simulation environment through a unified interface.
classdef environmentClass

    properties  %public properties
        % -- DEFAULT ENVIRONMENT PARAMS
        xmin    = -50;
        xmax    = 150;
        ymin    = -50;
        ymax    = 150;
        zmin    = 0;
        zmax    = 20;
        fil_locs = [];  %stores the plume particle positions [x y z concentration]
        PLOTAZM = 45;   %azimuth and elevation of the view angle when plotting
        PLOTELV = 40;
        Fs      = 20;   %20 Hz simulation, i.e. dt=0.05s
        dt      = [];
        %The flow field serves as the velocity field used to compute the position updates of the plume particles, i.e. the flow velocity determines the direction and speed of the "pollutant particles".
        %The plume model (usually a particle model or a grid model) advances the positions of the pollutant particles according to the flow velocity, and adds a diffusion mechanism to simulate random particle diffusion.
        ff      = flowfieldClass.empty; %flowfieldClass object array, representing one or more flow fields.
        plume   = filamentClass.empty;  %filamentClass object array, simulating one or more plumes.
        planner = plannerClass.empty;   %plannerClass object array, storing the path planners. UAV + sensing + quant
        terrain = terrainClass.empty;   %terrainClass object, representing the terrain information (optional).
        az      = 0;                    %az, el view angles, used for 3D plotting.
        el      = 90;
        frames  = 0;                    %GIF animation frame counter.
        fn      = 'test.gif';
        Fs_data = 10;

        %ColorSettings = {'r.','g.','b.'};
		ColorSettings = {'r.','b.'};    %plotting color settings for the different plume points.
    end

    properties (Access = private)   %private properties
        % properties exclusive to environment class, counters, used for multiple plumes and multiple UAVs
        currentTime = 0;
        NumFF = 1;          %number of flow fields.
        NumPlumes = [1,1];  %number of plumes, two-dimensional size (rows and columns).
        NumPlanners = 0;    %number of planners.
    end

    methods

        % -- Constructor
        function obj = environmentClass(Sim_Params)

            %Supports initializing multiple flow fields and multiple plumes; if no parameters are passed in, a single default object is initialized.
            % -- Check for initializiation parameters
            if nargin == 1
                % -- ADDITIONAL PARAMS
                %ff_params = [];

                % -- ATTACH Flowfield CLASS OBJECTS:    %create a flow field from the passed-in simulation parameters
                if isfield(Sim_Params,'ff_params')
                    obj.NumFF = length(Sim_Params.ff_params);   %read the number of flow fields
                    for i = 1:obj.NumFF
                        obj.ff(i)      = flowfieldClass(Sim_Params.ff_params(i));     % Create flow field
                        fprintf('Flow Field(s) Initialized \n')
                    end
                else
                    obj.ff      = flowfieldClass();     % Create flow field
                    obj.NumFF = 1;
                    fprintf('Initialized Single Flow Field with Default Settings \n')
                end

                % -- ATTACH Filament CLASS OBJECTS: create one or more plumes
                if isfield(Sim_Params,'plume_params')
                    for i = 1:size(Sim_Params.plume_params,1)
                        for j = 1:size(Sim_Params.plume_params,2)
                            obj.plume(i,j)       = filamentClass(Sim_Params.plume_params(i,j));      % Create plume (In progress)
                        end
                        fprintf('Plume(s) Initialized \n')
                    end
                    obj.NumPlumes = size(Sim_Params.plume_params);
                else
                    obj.plume       = filamentClass();      % Create plume (In progress)
                    obj.NumPlumes = 1;
                    fprintf('Initialized Single Plume with Default Settings \n')
                end

                % -- ATTACH Planner CLASS OBJECTS:  create one or more planner objects
                if isfield(Sim_Params,'planner_params')
                    obj.NumPlanners = length(Sim_Params.planner_params);
                    for i = 1:obj.NumPlanners
                        obj.planner(i) = plannerClass(Sim_Params.planner_params(i));      % Create plume (In progress)
                    end
                    fprintf('Planner(s) Initialized \n')
                else
                    obj.planner       = plannerClass();
                    fprintf('Initialized Single Planner with Default Settings \n')
                end

                % -- OTHER CONFIGURATIONS
                %obj.Fs = obj.ff.Fs;

%             else
%                 obj.ff      = flowfieldClass();     % Create flow field
%                 fprintf('Initialized Single Flow Field with Default Settings \n')
%                 obj.plume       = filamentClass();      % Create plume (In progress)
%                 fprintf('Initialized Single Plume with Default Settings \n')
%                 obj.planner       = plannerClass();      % Create plume (In progress)
%                 fprintf('Initialized Single Planner with Default Settings \n')
            end
            obj.dt = 1/obj.Fs;
        end

        %simulate, simulate_v2, simulateApp: provide different versions of the simulation-loop execution interface, supporting the command line and GUI applications.
        % -- Simulate functions
        function simulate(obj)
            for Plume_Sims = 1:obj.NumPlumes(1) %Plumes per simulation
                %Temporary Flowfields/Plumes to be overwritten during
                %loops. Ensures no continuation of results from prior sims.
                Temp.ff = obj.ff;
                Temp.PLUMES = obj.plume(Plume_Sims,:);

                ColorSettings = {'r.','g.','b.'};

                for FF_Per_Plume = 1:obj.NumFF %Flowfiels per plume
                    x = linspace(Temp.ff(FF_Per_Plume).xmin,Temp.ff(FF_Per_Plume).xmax,Temp.ff(FF_Per_Plume).n);
                    y = linspace(Temp.ff(FF_Per_Plume).ymin,Temp.ff(FF_Per_Plume).ymax,Temp.ff(FF_Per_Plume).m);
                    [xx,yy]=meshgrid(x,y);
                    for Frames = 1:1000 %Simulation Run
                        [Temp.ff(FF_Per_Plume), ~] = Temp.ff(FF_Per_Plume).step;
                        for Plumes_Per_FF = 1:obj.NumPlumes(2) %Plumes per flowfield
                            [Temp.PLUMES(Plumes_Per_FF), ~] = Temp.PLUMES(Plumes_Per_FF).step(Temp.ff(FF_Per_Plume));
                        end

                        %TEMPORARY PLOTTING
                        if mod(Frames,10)==0
                            quiver(xx,yy, Temp.ff(FF_Per_Plume).uv(:,:,1), Temp.ff(FF_Per_Plume).uv(:,:,2))
                            hold on
                            for i = 1:obj.NumPlumes(2)
                                scatter(Temp.PLUMES(i).fil(:,1),Temp.PLUMES(i).fil(:,2),ColorSettings{i})
                            end
                            hold off
                            axis([ Temp.ff(FF_Per_Plume).xmin, Temp.ff(FF_Per_Plume).xmax, Temp.ff(FF_Per_Plume).ymin, Temp.ff(FF_Per_Plume).ymax])
                            pause(0.01)

                        end

                    end
                end
                Temp.ff = [];
            end
        end

        % -- Alternative simulation
        function obj = simulate_v2(obj,params)
            if nargin == 1
                params.N = 1000;
                params.modnum = 10;
            end

            for k = 1: params.N
                obj = obj.step;
                if mod(k,params.modnum)==0
                    clf
                    if ~isempty(obj.terrain)
                        obj.plotterrain(1)
                    end
                    obj.plotff(1)
                    obj.plotplume(1)
                    shg
                end
            end
        end

        % -- Simulation version for GUI application
        function simulateApp(obj,app)
            for Plume_Sims = 1:obj.NumPlumes(1) %Plumes per simulation
                %Temporary Flowfields/Plumes to be overwritten during
                %loops. Ensures no continuation of results from prior sims.
                Temp.ff = obj.ff;
                Temp.PLUMES = obj.PLUMES(Plume_Sims,:);

                ColorSettings = {'r.','g.','b.'};

                for FF_Per_Plume = 1:obj.NumFF %Flowfiels per plume
                    x = linspace(Temp.ff(FF_Per_Plume).xmin,Temp.ff(FF_Per_Plume).xmax,Temp.ff(FF_Per_Plume).n);
                    y = linspace(Temp.ff(FF_Per_Plume).ymin,Temp.ff(FF_Per_Plume).ymax,Temp.ff(FF_Per_Plume).m);
                    [xx,yy]=meshgrid(x,y);
                    for Frames = 1:1000 %Simulation Run
                        [Temp.ff(FF_Per_Plume), ~] = Temp.ff(FF_Per_Plume).step;
                        for Plumes_Per_FF = 1:obj.NumPlumes(2) %Plumes per flowfield
                            [Temp.PLUMES(Plumes_Per_FF).plume, ~] = Temp.PLUMES(Plumes_Per_FF).plume.step(Temp.ff(FF_Per_Plume));
                        end

                        %TEMPORARY PLOTTING
                        if mod(Frames,10)==0
                            quiver(app.UIAxes,xx,yy, Temp.ff(FF_Per_Plume).uv(:,:,1), Temp.ff(FF_Per_Plume).uv(:,:,2))
                            hold(app.UIAxes,'on')
                            for i = 1:obj.NumPlumes(2)
                                scatter(app.UIAxes,Temp.PLUMES(i).plume.fil(:,1),Temp.PLUMES(i).plume.fil(:,2),ColorSettings{i})
                            end
                            hold(app.UIAxes,'off')
                            axis([ Temp.ff(FF_Per_Plume).xmin, Temp.ff(FF_Per_Plume).xmax, Temp.ff(FF_Per_Plume).ymin, Temp.ff(FF_Per_Plume).ymax])
                            pause(0.01)

                        end

                    end
                end
                Temp.ff = [];
            end
        end

        % -- Environment Step Function
        %First update the flow field state, then use the flow velocity to drive the motion of the plume particles.
        function obj = step(obj)
            % -- step flowfield independently
            % Call the step method of all flow field objects. 1 update the wind field  %2 update the plume particles  %3 UAV movement + sensing + quant  %4 save the plume data
            for i=1:length(obj.ff)  %call step for every flowfield to update the wind speed
                [obj.ff(i),~] = obj.ff(i).step;
            end
			% -- step filament(s) independently maintains the plume particles; the step method internally uses the flow velocity data to compute the change in particle positions;
            for j=1:size(obj.plume,2)
                if ~isempty(obj.terrain)    %if there is terrain, the plume is affected by the terrain
                    %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i),obj.terrain);
					[obj.plume(j),~] = obj.plume(j).step(obj.ff(i),obj.terrain);
                else    %otherwise the plume is only affected by the wind field
                   %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i));
					[obj.plume(j),~] = obj.plume(j).step(obj.ff(i));
                end
            end
			% for i=1:length(obj.ff)
                % [obj.ff(i),~] = obj.ff(i).step;
                % % -- step filament
                % for j=1:size(obj.plume,2)
                    % if ~isempty(obj.terrain)
                        % %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i),obj.terrain);
						% [obj.plume(j),~] = obj.plume(j).step(obj.ff(i),obj.terrain);
                    % else
                        % %[obj.plume(i,j),~] = obj.plume(i,j).step(obj.ff(i));
						% [obj.plume(j),~] = obj.plume(j).step(obj.ff(i));
                    % end
                % end
            % end

            % -- step planner
            %Call the step method of all path planners to update the motion and sampling strategy of the sampling equipment such as UAVs.
            % 1 UAV movement  % 2 sample concentration   % 3 sample wind speed  % 4 store data % 5 quantification (NGI/MB)
            for i=1:length(obj.planner)
                % main code here
                obj.planner(i) = obj.planner(i).step(obj.ff,obj.plume);
            end

            %Update fil_locs, storing the position and intensity information of all current plume particles.
            for i = 1:obj.NumPlumes(1)
                if i==2 && ~holdtrue, hold on; end
                if obj.ff(1).dim == 3 % if it is a 3D flowfield, save x y z concentration
                    %If the dimension of flowfield is 3, then the data will be 4 dimension, 3 dimension of wind(u,v,w), 1 dimension for methane concentration
                    obj.fil_locs = obj.plume(i).fil(:,[1:4]);
                else                    %if it is 2D, save x y concentration
                    obj.fil_locs = obj.plume(i).fil(:,[1,2,4]);
                end
            end

        end

        %addPlume, removePlume, addFlowfield, removeFlowfield, addPlanner, removePlanner manage the addition and removal of plume, flow field and planner objects in the environment, making it easy to adjust the simulation content dynamically.
        % -- Add plume(s) to the environment class properties
        function obj = addPlume(obj,pobj)
            len = length(obj.plume);
            if nargin == 1
                % -- initialize default to the environment class
                obj.plume(len+1) = filamentClass();
            elseif nargin == 2
                % -- initialize an add to the environment class
				for i=1:size(pobj,2)
					obj.plume(len+i) = pobj(i);
				end
            end
            obj.NumPlumes = length(obj.plume);
        end

        % -- Remove plume(s) in the environment class
        function obj = removePlume(obj,id)
            obj.plume(id) = [];
            obj.NumPlumes = size(obj.plume);
        end

        % -- Add flowfield(s) to the environment class properties
        function obj = addFlowfield(obj,ffobj)
            len = length(obj.ff);
            if nargin == 1
                % -- initialize default to the environment class
                obj.ff(len+1) = flowfieldClass();
            elseif nargin == 2
                % -- initialize an add to the environment class
                obj.ff(len+1) = ffobj;
            end
            obj.NumFF = len+1;
        end

        % -- Remove flowfield(s) in the environment class
        function obj = removeFlowfield(obj,id)
            obj.ff(id) = [];
            obj.NumFF = length(obj.ff);
        end

        % -- Add planners(s) to the environment class properties
        function obj = addPlanner(obj,plobj)
            len = length(obj.planner);
            if nargin == 1
                % -- initialize default to the environment class
                obj.planner(len+1) = plannerClass();
            elseif nargin == 2
                % -- initialize an add to the environment class
                obj.planner(len+1) = plobj;
            end
            obj.NumPlanners = len+1;
        end

        % -- Remove planner(s) in the environment class
        function obj = removePlanner(obj,id)
            obj.planner(id) = [];
            obj.NumPlanners = length(obj.planner);
        end

        %This is not used in our method, just used in the compared method fluxotaxis
        function obj = addTerrain(obj,sobj_params)
            if nargin == 2
                obj.terrain = terrainClass(sobj_params);
            else
                obj.terrain = terrainClass();
            end
        end
        % --Plotting functions  various plotting functions
        %Plot a 2D wind-field velocity vector map to show the flow field effect.
        function plotWind(obj)
            % -- get mesh for plotting
            x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
            y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            [xx,yy]=meshgrid(x,y);

            % -- plot the wind over some time
            figure(1)
            quiver(xx,yy,obj.ff.uv(:,:,1),obj.ff.uv(:,:,2))
            for i=1:obj.ff.Fs*200
                [obj.ff,flag] = obj.ff.step();
                if mod(i,20)==0
                    quiver(xx,yy,obj.ff.uv(:,:,1),obj.ff.uv(:,:,2))
                    %axis([obj.ff.xmin,obj.ff.xmax,obj.ff.ymin,obj.ff.ymax])
                    axis([obj.xmin,obj.xmax,obj.ymin,obj.ymax])
                    %view(gca,obj.PLOTAZM+25*cos(0.005*i),obj.PLOTELV+3*cos(0.01*i))
                    view(gca,obj.PLOTAZM,obj.PLOTELV)
                    text(0,0,1,num2str(i/obj.ff.Fs))
                    shg
                    pause(0.01)
                end

            end
        end

        % --Plotting functions  Plot the current plume particle distribution (not completed in the code).
        function plotPlume(obj)
            x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
            y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            [xx,yy]=meshgrid(x,y);


        end

        % --To plot all or individual flowfield at current frame plot the flow velocity field, 2D or 3D supported.
        function plotff(obj,holdtrue,ffnum,ffcolor)
            % -- get mesh for plotting
            x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
            y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            [xx,yy]=meshgrid(x,y);
            xx = xx'; yy = yy';
            c = 'k';
            if holdtrue, hold on; end
            if nargin >= 3
                i = ffnum;
                if obj.ff(i).dim == 3
                    uv_temp = squeeze(obj.ff(i).uv(1,:,:,1:2));
                    h = obj.ff.zmin;
                else
                    uv_temp = obj.ff(i).uv;
                    h = 0;
                end
				if nargin > 3
					%quiver(xx,yy,uv_temp(:,:,1),uv_temp(:,:,2),ffcolor)
                    quiver3(xx,yy,h+0*yy,uv_temp(:,:,1),uv_temp(:,:,2),0*yy,ffcolor)
				else
					quiver(xx,yy,uv_temp(:,:,1),uv_temp(:,:,2),c)
				end
            else
                for i=1:length(obj.ff)
                    if i==2 && ~holdtrue, hold on; end
                    if obj.ff(i).dim == 3
                        uv_temp = squeeze(obj.ff(i).uv(1,:,:,1:2));
                    else
                        uv_temp = obj.ff(i).uv;
                    end
                    quiver3(xx,yy,0*yy,uv_temp(:,:,1),uv_temp(:,:,2),0*yy,c)
                end
            end
            hold off
            axis([ obj.ff(1).xmin, obj.ff(1).xmax, obj.ff(1).ymin, obj.ff(1).ymax])

        end

        % -- To plot all of individual plumes at current frame plot the plume particles, supporting multiple plumes, multiple colors, and 3D or 2D scatter plots.
        function plotplume(obj,holdtrue,plumenum)
            % -- get mesh for plotting
            %x = linspace(obj.ff.xmin,obj.ff.xmax,obj.ff.n);
           % y = linspace(obj.ff.ymin,obj.ff.ymax,obj.ff.m);
            %[xx,yy]=meshgrid(x,y);
            if holdtrue, hold on; end
            psize = 50;
            if nargin == 3
                %i = plumenum;
                %if obj.ff(1).dim == 3
                    %scatter3(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.plume(i).fil(:,3),psize,obj.ColorSettings{i})
                    %view(obj.az,obj.el)
                %else
                    %scatter(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.ColorSettings{i})
                %end
				for i = 1:obj.NumPlumes(1)
                    if i==2 && ~holdtrue, hold on; end
                    if obj.ff(1).dim == 3
                        scatter3(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.plume(i).fil(:,3),psize,obj.ColorSettings{plumenum})
                        scatter3(obj.plume(i).p0(1),obj.plume(i).p0(2),obj.plume(i).p0(3),'y*')
                        view(obj.az,obj.el)
                    else
                        scatter(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.ColorSettings{plumenum})
                    end
                end
            else
                for i = 1:obj.NumPlumes(1)
                    if i==2 && ~holdtrue, hold on; end
                    if obj.ff(1).dim == 3
                        scatter3(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.plume(i).fil(:,3),500*psize*obj.plume(i).fil(:,4),obj.ColorSettings{1})
                        scatter3(obj.plume(i).p0(1),obj.plume(i).p0(2),obj.plume(i).p0(3),'g*')
                        view(obj.az,obj.el)
                    else
                        scatter(obj.plume(i).fil(:,1),obj.plume(i).fil(:,2),obj.ColorSettings{1})
                    end
                end
            end
			%alpha(0.5)
            hold off
            if obj.ff(1).dim == 3
                axis([ obj.ff(1).xmin, obj.ff(1).xmax, obj.ff(1).ymin, obj.ff(1).ymax, obj.ff(1).zmin, obj.ff(1).zmax*1.25])
                view(obj.az,obj.el)
            else
                axis([ obj.ff(1).xmin, obj.ff(1).xmax, obj.ff(1).ymin, obj.ff(1).ymax])
            end
        end

        %Plot the terrain surface, supporting 3D lighting and transparency.
        function plotterrain(obj,holdtrue)
            if holdtrue, hold on; end
            X = obj.terrain.surface.XData;
            Y = obj.terrain.surface.YData;
            Z = obj.terrain.surface.ZData;

            colormap gray
            surf(X,Y,Z);
            shading interp;                  % Interpolate color across faces.
            camlight headlight; %left, right                   % Add a light over to the left somewhere.
            lighting gouraud;                % Use decent lighting.
            axis equal vis3d;                % Set aspect ratio.
            %axis image
            xlabel('X')
            ylabel('Y')
            zlabel('Z')
            alpha(gca,0.5)
        end

        %Plot the vehicle positions and the sampling point distribution in the planner.
        function plotplanner(obj,holdtrue)
            if holdtrue, hold on; end
            Xg = obj.planner.Xg;
            c = obj.planner.qqm;
            n = obj.planner.np;
            for i=1: length(obj.planner.vehicle)
                p = obj.planner.vehicle(i).p;
                s = size(p);
                if s(2) == 2
                    scatter(p(1),p(2),'gd')
                else
                    scatter3(p(1),p(2),p(3),'gd','filled')
                end
            end
            cmax = max(c);
            cmin = min(c);
            cb = 1.98;
            cs = (c-cmin)/(cmax-cmin);
            cs(cs>1) = 1;
            k = (cs>0.5);
            if cmax == 0, cmax = 1; end
            scatter3(Xg(k,1),Xg(k,2),Xg(k,3),'.','CData',[1-cs(k),1-cs(k),1-cs(k)])
%             c(c<=0)=1e-15;
%             cmin = min(c);
%             cmax = max(c);
%             if cmax > 0 && cmax ~= cmin
%                 scatter3(Xg(:,1),Xg(:,2),Xg(:,3),(c-cmin)./(cmax-cmin)+1e-15)
%             else
%                 scatter3(Xg(:,1),Xg(:,2),Xg(:,3),c)
%             end
            hold off
%             alpha(0.5)
        end

        %Grab a frame from the current figure window and save it as a GIF animation.
        function obj = getFrame(obj)
            frame = getframe();
            obj.frames = obj.frames + 1;
            im = frame2im(frame);
            [imind,cm] = rgb2ind(im,256);
            if obj.frames == 1
                imwrite(imind,cm,obj.fn,'gif', 'Loopcount',inf);
            else
                imwrite(imind,cm,obj.fn,'gif','WriteMode','append');
            end
        end
           
    end



    methods (Access = private)

    end

end