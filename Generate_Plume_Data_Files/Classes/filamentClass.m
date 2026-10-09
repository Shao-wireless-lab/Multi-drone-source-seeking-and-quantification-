%% Filament Class
% ----------------------
% by Derek Hollenbeck
% 
% ...
%
% EXAMPLE:
%   ...
%Simulates the motion and diffusion of the multiple filament groups (filaments) formed after a gas (or other substance) is released.
%A filament carries information such as position, size (diffusion radius) and lifetime.
%Combined with the flow field (flowfieldClass) to simulate the effect of wind speed and dynamically advance the filament positions.
%Takes physical effects such as temperature inversion and collisions into account.
%Provides a measurement interface to compute the concentration (ppm) at any sensor.
% ----------------------

classdef filamentClass

    properties
        % -- DEFAULT Filament Constants
%        Nmax            = 1500;          % max filaments maximum number of filaments, limits the computation scale
        Nmax            = 15000;          % TODO: increased from 1500 to 15000
        N               = 150;           % number of filaments per puff initial number of filaments in each "puff"
        p0              = [5,5];        % source location
        Rsq0            = 0.01;         % initial filament size initial filament size (squared diffusion radius)
        fil             = [];           % filament position and size filament state matrix, each row represents one filament, containing position and size
        fil_t           = [];           % filament age 	filament "age", i.e. the time it has been alive
        dt              = [];
        Fs              = 20;           %sampling frequency (Hz)
        k               = 0;            %counter, used to add new filaments at regular intervals
        sigma           = 2;            %standard deviation of the position noise, simulating random diffusion (possibly one of the two sigmas in the diffusion model? the one in reference [28]?)
        gamma           = 0.001;        %size growth rate, used to simulate the diffusive growth of the filaments
        Q               = 0.00005225;   %kg/s source strength

        % -- DEFAULT Concentration Constants
        Tamb            = 300;             %True Ambient Temp.  ambient temperature and initial filament temperature
        T0              = 273;             %Filament Initial Temp.
        Na              = 6.02214076*10^23; %Avagadro's Number (6.0221409), i.e. k
        T               = 288;             %Measured Average Ambient Temp.
        P               = 101325;          %Ambient Pressure
        R               = 8.31446;         %Methane Gas Constant
        M               = 16.04;           %Molar mass [g/mol] (DEFAULT - Methane)
        zz = [];
        airNumberDensity= [];
        % -- DEFAULT Concentration Filter Constants (5)(6)parameters of the low-pass filter
        alpha           = 0.1;      %bandwidth
%        tau             = 10^-4;    %threshold
        tau             = 0;    %threshold, temporarily changed to 0

        % -- DEFAULT Temperature Inversion Constants
        TempInversion   = 'True';      %Temperature Inversion (3D only)
        k_t             = 0.5;
        k_b             = 0.06;


        % -- DEFAULT Temperature Inversion Constants
        Collisions      = 'True';
        CollisionMethod = 'V2'; %V1 or V2
    end

    properties (Access = private)
        % -- DEFAULT PRIVATE PARAMS AND VARIABLES
        C_old           = 0;
    end

    methods
        function obj = filamentClass(params)
            if nargin == 1
                % Filament Constants
                if isfield(params,'Nmax'),  obj.Nmax    = params.Nmax;  end
                if isfield(params,'N'),     obj.N       = params.N;     end
                if isfield(params,'p0'),    obj.p0      = params.p0;    end
                if isfield(params,'Rsq0'),  obj.Rsq0    = params.Rsq0;  end
                if isfield(params,'Fs'),    obj.Fs      = params.Fs;    end
                if isfield(params,'k'),     obj.k       = params.k;     end
                if isfield(params,'sigma'), obj.sigma   = params.sigma; end
                if isfield(params,'gamma'), obj.gamma   = params.gamma; end
                if isfield(params,'Q'),     obj.Q       = params.Q;     end
                % Concentration Constants
                if isfield(params,'Tamb'),  obj.Tamb    = params.Tamb;  end
                if isfield(params,'T0'),    obj.T0      = params.T0;    end
                if isfield(params,'Na'),    obj.Na      = params.Na;    end
                if isfield(params,'T'),     obj.T       = params.T;     end
                if isfield(params,'P'),     obj.P       = params.P;     end
                if isfield(params,'R'),     obj.R       = params.R;     end
                % Concentration Filter Constants
                if isfield(params,'alpha'), obj.alpha   = params.alpha; end
                if isfield(params,'tau'),   obj.tau     = params.tau;   end
                % Temperature Inversion Constants
                if isfield(params,'TempInversion'), obj.TempInversion = ...
                                                  params.TempInversion; end
                if isfield(params,'k_t'),   obj.k_t     = params.k_t;   end
                if isfield(params,'k_b'),   obj.k_b     = params.k_b;   end
                % Collisions
                if isfield(params,'Collisions'),    obj.Collisions =...
                                                  params.Collisions;    end
                if isfield(params,'CollisionMethod'),    obj.CollisionMethod =...
                                                  params.CollisionMethod;    end
            end
            %Q was originally in kg/s; dividing by the mass of a single molecule turns it into mol/s, i.e. how many mol of CH4 molecules are released per second; dividing again by (number of filaments per puff) gives the number of molecules per filament, in units of (molecules/s)
%            obj.Q             = (obj.Q / (2.656*10^-26))/ obj.N;    %source strength (unit: molecules/s): divide by the molecular mass and the number of filaments to normalize the release strength
            % TODO: changed to
            obj.Q = obj.Q / obj.N;   % directly kg / filament (no longer converted to molecules)

            obj.airNumberDensity    = obj.P*obj.Na/obj.T/obj.R; %this and zz below are both N0; not sure why it is computed twice. It is the number density of air molecules (molecules/m³). It can be used to compute the ppm unit: as used in the measurement functions below, dividing the "molecule number density" by the "air molecule number density airNumberDensity" and multiplying by 1e6 gives exactly ppm (volume fraction, Parts Per Million; 1 ppm means 1 cubic centimeter of pollutant per cubic meter of air)
            %N0 is the density of methane gas
            obj.zz = obj.P*obj.Na/obj.T/obj.R;  %N0 density of methane gas
            obj.dt                  = 1/obj.Fs; % time step
            obj.fil                 = ones(obj.N,1) * [obj.p0,obj.Rsq0];    %filament state matrix, each row represents one filament, containing position and size
            obj.fil_t               = zeros(obj.N,1);                       %filament "age", i.e. the time it has been alive

        end

        %Purpose: advance the state of every filament by one time step.
        %Parameters:   %ff: flow field object, provides wind speed information.
                 %sobj: simulation environment object, provides auxiliary functions such as obstacles and collision detection.
        function [obj,flag] = step(obj,ff,sobj)
            for i = 1:length(obj)
                obj(i).k = obj(i).k + 1;
                flag = 0;


                % -- add filaments if 1 sec has passed
                %Based on the time counter k, add a batch of new filaments every 1 second.
                if mod(obj(i).k, obj(i).Fs) == 0
%                    obj(i).fil = [obj(i).fil; ones(ff.n, 1) * [obj(i).p0 obj(i).Rsq0]]; % row vector is position p, R is the size (diffusion radius?)
%                    obj(i).fil_t = [obj(i).fil_t; zeros(ff.n, 1)];  %filament age

                    %TODO: changed ff.n to obj(i).N
                    obj(i).fil = [obj(i).fil; ones(obj(i).N, 1) * [obj(i).p0 obj(i).Rsq0]]; % row vector is position p, R is the size (diffusion radius?)
                    obj(i).fil_t = [obj(i).fil_t; zeros(obj(i).N, 1)];  %filament age
                end

                % -- get flowfield dimensions
                num_fil = size(obj(i).fil,1);

                e = ones(num_fil, 1);

                % obstacle collision mask
                %if isfield(flowfield_props, 'mask')
                %    mask = flowfield_props.mask;
                %end

                %Query the wind speed at the current filament positions from the flow field, and update the filament positions and sizes by combining random noise and the size-growth model.
                %The result is a num_fil × ff.dim matrix, each row being a filament's position and velocity vector
                v_a = ff.wind_measurement(obj(i).fil(:, 1:ff.dim));


                %R = sqrt(obj(i).Rsq0^2 + gamma*mod(obj(i).k, obj(i).Fs))
                %e = (e./2).*obj(i).fil;

                fil_I(:, 1:ff.dim) = obj(i).fil(:, 1:ff.dim); % initial position

                %Update the filament state matrix; each row represents one filament, containing the filament's position (x,y,z) and size (Rsq, i.e. the squared diffusion radius)
                %v_a is the wind speed read; randn(num_fil, ff.dim) generates a num_fil × ff.dim matrix of standard normal random numbers, simulating the randomness of diffusion (Brownian motion). It is multiplied by sigma (diffusion strength) to serve as the random perturbation.
                %obj(i).gamma*e generates a num_fil × 1 column vector representing the size growth of each filament (the change of the squared diffusion radius Rsq). The filament volume / diffusion area grows over time, and gamma is the growth rate.
                %Velocity of each filament = flow velocity + random diffusion velocity
                %[v_a + obj(i).sigma .* randn(num_fil, ff.dim)] (position part) is concatenated with [obj(i).gamma*e] (the squared diffusion radius) to form a num_fil × (ff.dim+1) matrix
                %All increments are multiplied by the time step to obtain the actual change in each filament's position and size within one time step.
                %Filament position (x,y or x,y,z) update: old position + velocity×dt (flow field + random diffusion); this is where flowfield and filament are coupled, the wind speed pushing the filaments to move
                %Filament size (Rsq) update: old size + gamma×dt
                obj(i).fil = obj(i).fil + [v_a + obj(i).sigma .* randn(num_fil, ff.dim) obj(i).gamma*e] * obj(i).dt;

                % Temperature Inversion Dynamics
                %If 3D and temperature inversion is enabled, correct the filament height position (z coordinate).
                if ff.dim == 3 && strcmp(obj(i).TempInversion,'True')
                    deltaT = (obj(i).T0 - obj(i).Tamb) * exp(-obj(i).k_t * obj(i).fil_t);
                    %vertical velocity
                    v_b = -obj(i).k_b * deltaT + 0.05;
                    %update z
                    obj(i).fil(:,3) = obj(i).fil(:,3) + v_b*obj(i).dt;
                end

                % 3D Collision Dynamics
                %If collision detection is enabled, call collision_measurement to detect and correct the filament trajectory after a collision.
                if ff.dim == 3 && strcmp(obj(i).Collisions,'True')
                    Collision = sobj.collision_measurement(obj(i).fil);
                    %Collision2 = sobj.collision_measurement2(fil_I,obj.fil,ff);


                    %disp(sum(Collision ~= Collision2))
                    ID_Collision = (Collision == 1);

                     if sum(ID_Collision)~=0 && strcmp(obj.CollisionMethod,'V2')

                        obj(i).fil(ID_Collision, 1:ff.dim) =...
                            Surface_Collision_NEW(obj(i),fil_I(ID_Collision, 1:ff.dim),...
                            obj(i).fil(ID_Collision, 1:ff.dim),ff,sobj);
    %                     obj.fil(ID_Collision, 1:ff.dim) =...
    %                         Surface_Collision(obj,fil_I(ID_Collision, 1:ff.dim),...
    %                         obj.fil(ID_Collision, 1:ff.dim),sobj);
                     end


                end
                fil_I = [];
                % Aging of Filament update the filament age.
                obj(i).fil_t  = obj(i).fil_t  + obj(i).dt;

                % Filament Boundary Conditions
                %Filter out the filaments beyond the simulation space boundary: find the indices of all out-of-bound ones and then empty them
                if ff.dim == 2
                    %out_of_bounds = [ % indices of filaments which are out of bound
                        %find(obj(i).fil(:, 1) > ff.m * ff.dx | obj(i).fil(:, 1) < 0);...%dx);...
                        %find(obj(i).fil(:, 2) > ff.n * ff.dy | obj(i).fil(:, 2) < 0);...%dy)...
                        %];
                    out_of_bounds = [ % indices of filaments which are out of bound
                        find(obj(i).fil(:, 1) > ff.xmax | obj(i).fil(:, 1) < ff.xmin);...%dx);...
                        find(obj(i).fil(:, 2) > ff.ymax | obj(i).fil(:, 2) < ff.xmax);...%dy)...
                        ];
                elseif ff.dim == 3
                    out_of_bounds = [ % indices of filaments which are out of bound
                        find(obj(i).fil(:, 1) > ff.xmax | obj(i).fil(:, 1) < ff.xmin);...%dx);...
                        find(obj(i).fil(:, 2) > ff.ymax | obj(i).fil(:, 2) < ff.ymin);...%dy);...
                        find(obj(i).fil(:, 3) > ff.zmax | obj(i).fil(:, 3) < ff.zmin)...
                        ];
                end
                %delete the out-of-bound filaments
                obj(i).fil(out_of_bounds, :) = [];
                obj(i).fil_t (out_of_bounds) = [];
                num_fil = size(obj(i).fil, 1);

                %Limit the total number of filaments to at most Nmax; if exceeded, keep only the newest filaments.
				if (num_fil > obj(i).Nmax)
                %if isfield(obj(i), 'Nmax') && (num_fil > obj(i).Nmax)
                    obj(i).fil = obj(i).fil(end-obj(i).Nmax + 1:end, :);
                    obj(i).fil_t = obj(i).fil_t(end-obj(i).Nmax + 1:end, :);
                end


            end
        end

        %Compute the concentration at the given sensor positions, based on the diffusion radii and positions of all filaments, superposed with a Gaussian model.
        %Internally implements concentration smoothing filtering that removes low-concentration noise (concentrations below the threshold tau are set to zero).
        function C = concentration_measurement(obj,sensors)
            for i = 1:length(obj)
                %size(obj(i).fil); % get's size of filaments
                dim = size(obj(i).fil,2) - 1; % dimension of position vector

                p = obj(i).fil(:, 1:dim);
                Rsq = obj(i).fil(:, end);

                %Eq. (4)
                C{i} = [];
                for x = sensors.' % loop through agents
                    r = vecnorm((p - x.').').'; % matrix of distance between agent and all filaments
                    C_x = obj(i).Q ./ sqrt(8 .* pi^3) ./ sqrt(Rsq).^3 .* exp(-r.^2 ./ (2*Rsq)); % matrix of concentration of each filament detected by agent
                    C{i} = [C{i} sum(C_x)]; % sums concentrations and appends to C list
                end
                %The conversion to ppm has been removed, keeping the original unit of Q
%                C{i} = C{i} * 10^6 / obj(i).airNumberDensity; % convert to ppm

                %concentration_measurement2 is what is actually used; this if-end has been removed there
                if  obj(i).airNumberDensity~= obj.zz
                    disp(zz)
                    disp(obj(i).airNumberDensity)
                    quit
                end

                %Low-pass filter, Eq. (5)
                C{i} = C{i} + obj(i).alpha * obj(i).dt * (C{i} - obj(i).C_old);
                C{i}(C{i} < obj(i).tau) = 0;
                obj(i).C_old      = C{i};
            end
            C = cell2mat(C);    %convert to a matrix
        end

        %Similar to the method above: compute the concentration from the sensors' readings and pass it through the low-pass filter to get the processed concentration
        function C = concentration_measurement2(obj,sensors)
            for i = 1:length(obj)
                %size(obj(i).fil); % get's size of filaments
                dim = size(obj(i).fil,2) - 1; % dimension of position vector, drop the last dimension (Rsq); what remains is the dimension of position: 3D=3, 2D=2

                p = obj(i).fil(:, 1:dim); % filament position
                Rsq = obj(i).fil(:, end); % filament size Rsq

                C{i} = [];
                for x = sensors.' % loop through agents
                    r = vecnorm((p - x.').').'; % matrix of distance between agent and all filaments
                    %TODO: not sure why Rsq is multiplied by 2 here rather than squared (^2); the paper uses ^2
                    C_x = obj(i).Q ./ sqrt(8 .* pi^3) ./ sqrt(Rsq).^3 .* exp(-r.^2 ./ (2.*Rsq)); % matrix of concentration of each filament detected by agent
                    %C_x = (obj(i).Q ./ (sqrt(8 .* pi^3) .* Rsq.^3)) .* exp(-r.^2 ./ (Rsq.^2)); % matrix of concentration of each filament detected by agent
                    C{i} = [C{i} sum(C_x)]; % sums concentrations and appends to C list  % superposition of all filaments
                end
                %removed, no conversion to ppm
%                C{i} = C{i} * 10^6 / obj(i).airNumberDensity; % convert to ppm
                %disp(obj(i).Q)

                %if C{i} > 0
                %    disp('check')
                %end

                %Low-pass filter, Eq. (5); alpha is the bandwidth B
                C{i} = C{i} + obj(i).alpha * obj(i).dt * (C{i} - obj(i).C_old);
                C{i}(C{i} < obj(i).tau) = 0;
                obj(i).C_old      = C{i};
            end
            C = cell2mat(C);
            %if C ~= 0
            %    disp('error')
            %end
        end

        %Project the filament positions onto the ground (z=0) and then compute the concentration at the sensors, suitable for 2D horizontal concentration estimation. (Never used; compared with concentration_measurement2 the main difference is that the third dimension of the filament state matrix is set to 0, i.e. z=0)
        function C = concentration_proj(obj,sensors)
            for i = 1:length(obj)
                %size(obj(i).fil); % get's size of filaments
                dim = size(obj(i).fil,2) - 1; % dimension of position vector    obj(i).fil is the filament state matrix, including position and size; subtracting the dimension of the position leaves the position

                p = obj(i).fil(:, 1:dim);
                Rsq = obj(i).fil(:, end);   %the last dimension of the filament state matrix is the size, i.e. the radius Rf

                % project p onto the floor  the 3rd-dimension position is simply dropped and set to 0
                p(:,3) = 0;

                C{i} = [];
                for x = sensors.' % loop through agents
                    r = vecnorm((p - x.').').'; % matrix of distance between agent and all filaments    the 2-norm in Eq. (4), the 2-norm of the relative position; why take the transpose three times??
                    C_x = obj(i).Q ./ sqrt(8 .* pi^3) ./ sqrt(Rsq).^3 .* exp(-r.^2 ./ (2*Rsq)); % matrix of concentration of each filament detected by agent
                    C{i} = [C{i} sum(C_x)]; % sums concentrations and appends to C list
                end
%                C{i} = C{i} * 10^6 / obj(i).airNumberDensity; % convert to ppm

                C{i} = C{i} + obj(i).alpha * obj(i).dt * (C{i} - obj(i).C_old);

                C{i}(C{i} < obj(i).tau) = 0;
                obj(i).C_old      = C{i};
            end
            C = cell2mat(C);
        end

    end

    %Private methods for collision handling: Surface_Collision_NEW and Surface_Collision
    %These two methods handle the collision of filaments with the terrain surface or obstacles, simulating an elastic collision rebound.
    methods (Access = private)
        function v3 = Surface_Collision_NEW(obj,v1,v2,ff,sobj)
            xid = round([v1(:,1) v2(:,1)]./ff.dx)+1;
            yid = round([v1(:,2) v2(:,2)]./ff.dy)+1;
            Z = sobj.surface.ZData(:);
            V(:,:,1) = [sobj.surface.XData(1,xid(:,1))',...
                        sobj.surface.YData(yid(:,1),1),...
                        Z(size(sobj.surface.ZData,1)*(xid(:,1)-1)+yid(:,1))];
            V(:,:,2) = [sobj.surface.XData(1,xid(:,1))',...
                        sobj.surface.YData(yid(:,2),1)+1./1000,...
                        Z(size(sobj.surface.ZData,1)*(xid(:,1)-1)+yid(:,2))];
            V(:,:,3) =[sobj.surface.XData(1,xid(:,2))'+1./1000,...
                        sobj.surface.YData(yid(:,1),1),...
                        Z(size(sobj.surface.ZData,1)*(xid(:,2)-1)+yid(:,1))];
            n = cross(V(:,:,1)-V(:,:,3),V(:,:,1)-V(:,:,2),2);
            n(V(:,2,2)<V(:,2,1) & V(:,1,3)>=V(:,1,1),:) = -1*n(V(:,2,2)<V(:,2,1) & V(:,1,3)>=V(:,1,1),:);
            n(V(:,2,2)>=V(:,2,1) & V(:,1,3)<V(:,1,1),:) = -1*n(V(:,2,2)>=V(:,2,1) & V(:,1,3)<V(:,1,1),:);
            n = n./vecnorm(n,2,2);
%             collided = sum((n.*(v2(:,1:3) - V(:,:,1))),2)<=0;
%             v3 = v2;
%             v3(collided,:) = v2(collided,:) + 2*( dot(V(collided,:,1)- v2(collided,:),n(collided,:),2)).*n(collided,:);
            v3 = v2 + 2*(dot(V(:,:,1)- v2,n,2)).*n;
            Collision = sobj.collision_measurement(v3);
            ID_Collision = (Collision == 1);
            if sum(ID_Collision)~=0
                v3(ID_Collision,:) = v1(ID_Collision,:);
            end
        end
        function [R2] = Surface_Collision(obj,a,b,sobj)
            %SURFACE_COLLISION - This function simulates an elastic collision between the
            %filaments and the specified terrain/surface

            %     a - Initial Position
            %     b - Final Position Without Obstacle
            %     s - Surface Structure (with X, Y, and Z data)
            %     Nxyz - Cell containing normal vectors at all points of surface, {Nx,Ny,Nz}

            %Extract X,Y,Z Data from surface structure
            X = sobj.surface.XData;
            Y = sobj.surface.YData;
            Z = sobj.surface.ZData;

            %Create a vector of discrete points between initial and final positions
            r1 = [a(:,1) a(:,2) a(:,3)]; % First point
            r2 = [b(:,1) b(:,2) b(:,3)]; % Second point without collision

            dr = r2-r1;  % Vector from r1 to r2
            L = vecnorm(dr,2,2); % Length between two points
            n = dr./L; % Unit vector along direction

            x = 0:0.1:1; % Vector for vector length discretization
            R = repelem(r1,length(x),1) + (repmat(x',size(r1,1),1)).*...
                repelem(L,length(x),1).*repelem(n,length(x),1); %Create the vector of points for every set of points

            %Find the point of colllision on the surface
            zproj = interp2(X,Y,Z,R(:,1),R(:,2)); % Project the discrete vector between r1 and r2 onto the surface and
            % obtain z values on surface at same xy-coords
            R(R(:,3)<zproj,3) = NaN; % Set points below surface to NaN
            id = []; % Initialize the collision id as empty
            xp = arrayfun( @(v) CollisionIndex(obj,R,x,v) , 0:size(r1,1)-1 ); % Use CollisionIndex function below for every set
            % points. Obtains index first occurance of NaN
            % for each vector of points
            Rint = R(xp,:); % Apply index found from previous command to R to get collision point for each set of r1's and r2's

            %Extract normal vectors from the surface and interpolate on surface to
            %find normal vector at each collision point
            Nxx = interp2(X,Y,sobj.Nxyz{1},Rint(:,1),Rint(:,2),'nearest');
            Nyy = interp2(X,Y,sobj.Nxyz{2},Rint(:,1),Rint(:,2),'nearest');
            Nzz = interp2(X,Y,sobj.Nxyz{3},Rint(:,1),Rint(:,2),'nearest');
            N = [Nxx Nyy Nzz]; % Organize affirmentioned interpolated normal vectors into a matrix

            %Obtain final position after collision (R2)
            dr = r2-Rint; % Array pointing from r2 towards R
            C = (dot(dr,N,2)./(sum(N.^2,2))).*N; % Project dr onto N to get orthogonal distance
            R2= r2 + -2.*C; % Add twice of orthogonal distance vector to r2 to get final position with obstacle

            %Check if R2 is valid - second collision?
            zproj0 = interp2(X,Y,Z,R2(:,1),R2(:,2)); % Project new positions onto surface
            id = find(double(R2(:,3)<zproj0)==1); % Collision id gives index wherever there is another collision

            %Secondary collision process if the collision id is not empty
            if isempty(id) ~= 1
                %Create sphere at every initial collision point (Rint) with
                %points of radius equal to distance between collision point and
                %final position (distance from Rint to R2
                r = vecnorm(R2(id,:)-Rint(id,:),2,2); % Obtain distance between
                [x,y,z] = sphere(5); % Create 1 sphere
                XYZ = repmat([x(:) y(:) z(:)],length(id),1); % Repeat sphere points for collision id
                Rint = repelem(Rint(id,:),length(x(:)),1);  % Repeat Rint for linear vectorized length of set
                                                            % of coords for 1 sphere. (linear vectorized
                                                            % length as in collapsing x,y, or z matrices into
                                                            % long vectors and taking that length)
                r = repelem(r,length(x(:)),1); % Repeat radius of spheres for same length as Rint
                XYZ = XYZ.*r + Rint;    % For each sphere of points, multiply points by corresponding radius and add them
                                        % by center point (Rint)
                                        %Remove sphere points below surface
                Zproj = interp2(X,Y,Z,XYZ(:,1),XYZ(:,2)); % For all sphere points, project them onto the surface
                XYZ = [XYZ repelem(id,length(x(:)),1)]; % Add collision ID to every point on sphere of points corresponding
                                                        % to collision id
                XYZ = XYZ(XYZ(:,3)>Zproj,:); % Remove all sphere points below surface

                [~,FillID2,~] = unique(XYZ(:,4),'last');    %All id points in 4th column are in numerical order and repeated (i.e.
                                                            %could be [1;1;1;2;3;3;3;3...]). For each (for same example aboce,
                                                            % [3,4,8...].
                FillID1 = [1 (FillID2(1:(end-1))+1)']; % Using FillID2, use end of id index intervals to find beginning of said intervals
                NewInd = floor((FillID2-FillID1').*rand(length(FillID1),1)+FillID1'); % Generate random numbers between found intervals - pick
                                                                                        %random point in each set of qualified points for each sphere
                R2(id,:) = XYZ(NewInd,1:3); % Make random new points the new final points
            end
        end

        %Helper that determines the index of the specific collision position of a filament along a given trajectory.
        function out = CollisionIndex(obj,R,x,v)
            %COLLISIONINDEX - Given a matrix R filled with sets of discretized points
            %between different r1 and r2 values stacked vertically, the third column
            %will have NaN values. This code leverages that to find intersection points
            %for every collision.
            if (sum(isnan(R((length(x)*v+1):(length(x)*(v+1)),3)))>0)
                %For any interval determined by v
                %={0,1,2...} where there is an NaN value(s), the index of the point right
                %before the NaN row(s) will be returned.
                out = find(isnan(R((length(x)*v+1):(length(x)*(v+1)),3)),1,'first')+(length(x)*v) - 1;
            else
                %If no NaN values, then index in end of interval is returned -
                %final initial R2 value calculated will be used
                out = length(x)*(v+1);
            end
        end
    end
    
end