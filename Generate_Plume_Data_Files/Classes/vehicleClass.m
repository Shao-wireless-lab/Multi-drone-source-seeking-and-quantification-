%% UAV Class
% ----------------------
% by Derek Hollenbeck
%
% This class creates an environment object that acts as a container for
% running the MOABS/DT simulation
%
% EXAMPLE:
%   ...
% Simulates the motion and control of a UAV or similar vehicle, mainly used as the mobile vehicle (e.g. UAV, ground vehicle) in environment simulation.
% ----------------------
classdef vehicleClass

    properties
        % default params and variables

        % -- DEFAULT UAV Constants
        p               = [25 25 10];   %   current position (x,y,z)
        v               = [0 0 0];      % current velocity vector
        C               = [];           %   concentration measurement (measured by the sensor)
        ws              = [];           %   wind speed measurement
        dt              = [];           % time step, in seconds
        Fs              = 20;           % sampling frequency (Hz)
        alim            = 1;    % acceleration limit of the control input
        vlim            = 2;    % maximum speed limit
        Kp              = 0.5+2;        % proportional controller gain for the position error
        Kd              = 1.65+0.2;     % derivative controller gain for the velocity error
        WaypointProx   = 0.2;           % waypoint proximity threshold, used to decide whether a waypoint is reached
        vehicleType     = 2; % 1:int, 2:dint, 3:uav, 4:ugv, 5: h-vtol   vehicle type, 1: single integrator, 2: double integrator, 3: UAV, 4: ground vehicle, etc.
        vehicleMode     = 2; % 1:standby, 2:auto    % operating mode, 1: standby, 2: automatic operation
        sensor          = sensorClass.empty;    % sensor object array (instances of sensorClass)
        QuantifyFlag    = 0;            % flag for whether to perform concentration quantification
    end

    properties (Access = private)
        e               = [];   % position error (difference between the current target position and the current pose)
        U               = [];   % control input (acceleration)
        Waypoint        = [];   % current target waypoint
        Waypoints       = [];   % list of all waypoints
        Waypoint_ID     = 1;    % index of the current target waypoint
        numOfWaypoints  = [];   % total number of waypoints
        numOfSensors    = 0;    % number of sensors
        Waypoint_flag   = 0;    % flag for whether the current waypoint has been reached
    end

    methods

        function obj = vehicleClass(params)
            if nargin == 1
                % UAV Constants
                if isfield(params,'p'),     obj.p       = params.p;     end
                if isfield(params,'v'),     obj.v       = params.v;     end
                if isfield(params,'dt'),    obj.dt      = params.dt;    end
                if isfield(params,'alim'),  obj.alim    = params.alim;  end
                if isfield(params,'vlim'),  obj.vlim    = params.vlim;  end
                if isfield(params,'Kp'),    obj.Kp      = params.Kp;    end
                if isfield(params,'Kd'),    obj.Kd      = params.Kd;    end
                if isfield(params,'WaypointProx')
                    obj.WaypointProx = params.WaypointProx; end
                if isfield(params,'Waypoint')
                    obj.WaypointList = [obj.WaypointList;params.Waypoint];
                    obj.Waypoint = obj.WaypointList(obj.WaypointNum);
                end
            end
            obj.dt                  = 1/obj.Fs; % time step
            obj.e                   = 0;    % initialize the error e=0 and the control input U=0.
            obj.U                   = zeros(size(obj.p));
            obj.sensor              = sensorClass();
            obj.sensor.eta          = 0.01; % sensor1, corresponds to the wind speed measurement
            obj.sensor(2)           = sensorClass();
            obj.sensor(2).ID        = 2;
            obj.sensor(2).eta       = 2.2336e-08;   % measurement noise term of the concentration sensor, set according to the SNR value.
%            obj.sensor(2).eta       = 0.005/1000;   % noise term of the concentration measurement of sensor2
            obj.sensor(2).alph      = 1;
            obj.numOfSensors        = length(obj.sensor);
        end

        % Mainly performs the vehicle dynamics and control update.
        function [obj, flag] = stepDynamics(obj)
            flag = 0;
            if obj.vehicleMode == 2 % auto  only runs in auto mode

                switch obj.vehicleType

                    case 1 % single integator

                    case 2 % double integrator (PD controller)
                        pi = obj.p;
                        vi = obj.v;
                        ei = obj.e; % the error at the previous time step
                        Ui = obj.U;
                        obj.e  = obj.Waypoint - pi;
                        obj.U = obj.Kp.*obj.e + obj.Kd.*(obj.e - ei)./obj.dt;   % classic PD control, the d term is the derivative of the error
                        % acceleration saturation
                        if vecnorm(obj.U,2)> obj.alim
                            obj.U = obj.U./vecnorm(obj.U,2) .* obj.alim;
                        end
                        % update velocity and position with a semi-implicit Euler method
                        % the velocity is updated from the acceleration
                        obj.v = vi + 0.5*obj.dt.*(obj.U + Ui);
                        if vecnorm(obj.v,2)> obj.vlim
                            obj.v = obj.v./vecnorm(obj.v,2) .* obj.vlim;
                        end
                        % the position is updated from the velocity, so there is no localization; the position is just obtained by integrating the velocity
                        obj.p   = pi + 0.5*obj.dt.*(obj.v+vi);

                        % -- Take sensor measurements
                        %obj.ws  = ff.wind_measurement(obj.p);
                        %obj.C   = filament.concentration_measurement(obj.p);


                        % -- Reached location   judge whether the waypoint is reached according to the proximity threshold; this is a return value of the function
                        flag = norm(obj.p - obj.Waypoint)<obj.WaypointProx;

                end
            end
            % monitor whether the waypoint has been reached
%            fprintf('Waypoint_flag or flag = %d, norm(obj.p - obj.Waypoint):=%.4f, obj.WaypointProx= %.2f\n',...
%                flag, norm(obj.p - obj.Waypoint), obj.WaypointProx);

            dist = norm(obj.p - obj.Waypoint);
            fprintf('obj.vehicleMode=%d | dflag=%d | dist=%.4f | prox=%.2f | p=[%.2f %.2f %.2f] | wp=[%.2f %.2f %.2f] | size(obj.Waypoints,1)=%d\n',...
            obj.vehicleMode, flag, dist, obj.WaypointProx, ...
            obj.p(1),obj.p(2),obj.p(3), ...
            obj.Waypoint(1),obj.Waypoint(2),obj.Waypoint(3), size(obj.Waypoints,1));

            % When the last waypoint is reached and it is marked as reached, QuantifyFlag is turned off.
            if (obj.Waypoint_ID == size(obj.Waypoints,1)) && (obj.Waypoint_flag == 1)
                obj.QuantifyFlag = 1; % set to 0 to not quantify, set to 1 to quantify after waypoint mission
                obj.Waypoint_ID = 1;
            end
        end

        function obj = step(obj)
            try
                obj.Waypoint = obj.Waypoints(obj.Waypoint_ID,:);    % read the waypoint corresponding to the next ID
                [obj, obj.Waypoint_flag]     = obj.stepDynamics();  % PD control makes the vehicle move

                %After reaching a current waypoint, start going to
                %next waypoint  after reaching the current waypoint, set the next waypoint as the target
                if (obj.Waypoint_flag == 1) && (obj.Waypoint_ID < size(obj.Waypoints,1))
                    obj.Waypoint_ID = obj.Waypoint_ID + 1;
                end

                %After passing the first waypoint, do all of this
                %                 if obj.Waypoint_ID > 1
                %
                %                     %Flag being ready to quantify once the final
                %                     %waypoint is reached
                %                     if (obj.Waypoint_ID ==...
                %                             size(obj.Waypoint,1)) && ...
                %                             (obj.Waypoint_flag == 1)
                %
                %                         obj.QuantifyFlag = 1;
                %                     else
                %                         % -- Store these values only when in the plane
                %                         obj.WindSpeed       = [obj.WindSpeed; obj.vehicle.ws];
                %                         obj.Position        = [obj.Position; obj.vehicle.p];
                %                         obj.Concentration   = [obj.Concentration; obj.vehicle.C];
                %                     end
                %                 end
                %                 %If at the final waypoint and Fe isn't calculates
                %                 if obj.QuantifyFlag == 1 && obj.Fe == []
                %                     obj = Quantify(obj, ff, filament);
                %                 end
            catch
                disp('Vehicle Error:')
            end
        end

        % -- get function   get the current waypoint (i.e. the current target point)
        function waypoint = getWaypoint(obj)
            waypoint = obj.Waypoint;
        end

        % -- get function   get all waypoints
        function waypoints = getWaypoints(obj)
            waypoints = obj.Waypoints;
        end

        % -- get function   get the ID of the current waypoint
        function waypoint_id = getWaypoint_ID(obj)
            waypoint_id = obj.Waypoint_ID;
        end

        % -- set function
        %         function obj = setWaypoint(obj,waypoint)
        %             obj.Waypoints = waypoint;
        %             obj.Waypoint_ID = 1;
        %             obj.Waypoint = obj.Waypoints(obj.Waypoint_ID);
        %         end

        % -- set function  set the waypoint list, automatically reset Waypoint_ID to 1, and set the current waypoint
        function obj = setWaypoints(obj,waypoints)
            obj.Waypoints = waypoints;  % accept the passed-in waypoints
            obj.Waypoint_ID = 1;
            obj.Waypoint = obj.Waypoints(obj.Waypoint_ID,:);
        end

        % -- set function  set the current waypoint index.
        function obj = setWaypoint_ID(obj,waypoint_id)
            obj.Waypoint_ID = waypoint_id;
        end

        % -- get Sensor reading functions given input: ff, plume, etc.
        %         function obj = getSensorReading(obj,type,input)
        %             switch type
        %                 case 'point source'
        %                     obj.C   = input.concentration_measurement(obj.p);
        %                 case 'wind'
        %                     obj.ws  = input.wind_measurement(obj.p);
        %             end
        %         end

        % Add sensors, batch adding is supported.
        function obj = addSensor(obj,sensors)
            % add multiple sensors
            if nargin == 2
                len = length(obj.sensor);
                for i=1:length(sensors)
                    params.type = sensors(i);
                    obj.sensor(len+i) = sensorClass(params);
                end
                obj.numOfSensors = length(obj.sensor);
            % add a single sensor
            else
                obj.sensor = sensorClass();
                obj.numOfSensors = 1;
            end
        end

        % get the current position
        function pos = getPostion(obj)
            pos = obj.p;
        end



    end

    methods (Access = private)

    end
end