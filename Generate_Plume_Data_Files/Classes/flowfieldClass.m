%% Flowfield Class
% ----------------------
% by Derek Hollenbeck
% 
% ...
%
% EXAMPLE:
%   ...
%Generates the wind-speed (uv) flow field of a 2D or 3D grid to simulate the ambient wind.
%Uses a stochastic process with colored noise to drive the dynamic variation of the flow field, supporting time-stepping updates.
%Supports measuring the wind speed through spatial coordinates (wind_measurement), providing environment-perception data for simulated objects such as UAVs.
%Provides management of the time state.
% ----------------------

classdef flowfieldClass

    properties
        % default params and variables
        K           = [1000,1000];  % diffusivity          % wind-field diffusion coefficient (used for solving the PDE)   % K(1) -> diffusion in the x direction   % K(2) -> diffusion in the y direction
        n           = 20;           % rows of flowfield 	grid size: number of rows (x direction), number of columns (y direction) and number of height layers (z direction); the range is 0-200, but it is gridded into 20 rows, so the step size needs to be computed
        m           = 20;           % columns of flowfield
        l           = 10;            % height of flowfield
        dim         = 2;            % dimension of flowfield  wind field dimension
        uv          = [];           % flow field
        %abG         = [0.01,0.01,5];% colored noise params
        abG         = [0.5,0.1,3];% colored noise params    colored noise parameters, controlling the statistical properties of the noise
        uv0         = [1,0];        % initial/mean wind direction
        tf_num      = [];           % transfer function numerator   numerator/denominator coefficients of the digital filter used to generate the colored noise, used to compute H(s)
        tf_den      = [];           % transfer function denominator
        xmin        = 0;            % min range in x-axis
        xmax        = 200;          % max range in x-axis
        ymin        = 0;            % min range in y-axis
        ymax        = 200;          % max range in y-axis
        zmin        = 0;            % min range in z-axis
        zmax        = 20;           % max range in z-axis
        Fs          = 20;           % sampling frequency
        cn          = [];           % colored noise buffer storing the colored noise
        wn          = [];           % white noise buffer for the white noise
        zf          = [];
        dx          = [];           %grid cell size
        dy          = [];
        dz          = [];
        dt          = [];           %time step, equal to 1/Fs
        alph        = 0.3;          %Power law scaling exponent exponent for amplitude scaling, used for the height attenuation of the 3D flow field
    end

    properties (Access = private)
        % properties exclusive to environment class
        currentTime = 0;            % private property, records the current simulation time
        uv_temp     = [];           % temporary flow field variable, used for the update computation
    end

    methods

        function obj = flowfieldClass(params)
            %Supports passing in a struct of parameters to override the default properties.
            if nargin == 1  %nargin is a built-in variable in MATLAB functions that indicates the number of arguments passed when calling the current function. ==1 means one argument (params) was passed when instantiating the object
                if isfield(params,'K'),     obj.K       = params.K;     end
                if isfield(params,'n'),     obj.n       = params.n;     end
                if isfield(params,'m'),     obj.m       = params.m;     end
                if isfield(params,'l'),     obj.l       = params.l;     end
                if isfield(params,'dim'),   obj.dim     = params.dim;   end
                if isfield(params,'abG'),   obj.abG     = params.abG;   end
                if isfield(params,'uv0'),   obj.uv0     = params.uv0;   end
                if isfield(params,'xmin'),  obj.xmin    = params.xmin;  end
                if isfield(params,'xmax'),  obj.xmax    = params.xmax;  end
                if isfield(params,'ymin'),  obj.ymin    = params.ymin;  end
                if isfield(params,'ymax'),  obj.ymax    = params.ymax;  end
                if isfield(params,'zmin'),  obj.zmin    = params.zmin;  end
                if isfield(params,'zmax'),  obj.zmax    = params.zmax;  end
                if isfield(params,'Fs'),    obj.Fs      = params.Fs;    end
                if isfield(params,'alph'),  obj.alph    = params.alph;  end
            end
            obj.dx = (obj.xmax-obj.xmin)/obj.n;% change in x-axis   compute the spatial step and the time step.
            obj.dy = (obj.ymax-obj.ymin)/obj.m;% change in y-axis
            obj.dz = (obj.zmax-obj.zmin)/obj.l;% change in z-axis
            obj.dt = 1/obj.Fs;         % time step



            N = [0,0,obj.abG(1)*obj.abG(3)];
            D = [1,obj.abG(2),obj.abG(1)];
            % -- Control toolbox Installed
            %dsys = c2d(tf(N,D),obj.dt);
            %obj.tf_num = cell2mat(dsys.Numerator);
            %obj.tf_den = cell2mat(dsys.Denominator);
            % -- No Control Toolbox installed
            T = 1/obj.Fs;
            a = obj.abG(1); b = obj.abG(2); G = obj.abG(3);
            n2 = a*G; n1 = 2*n2; n0 = n2;
            d2 = (4/T^2 + 2*b/T + a); d1 = (-8/T^2 + 2*a); d0 = (4/T^2 - 2*b/T + a);
            obj.tf_num = [n2/d2,n1/d2,n0/d2];   %colored noise process' transfer function numerator
            obj.tf_den = [1,d1/d2,d0/d2];       %denominator

            %uv is initialized to 0, representing the initial no-wind or still state.
            obj.uv_temp = zeros(obj.n,obj.m,obj.dim);
            obj.uv = zeros(obj.n,obj.m,obj.dim);    %For a 2D wind field there is no height variation. Note that the storage order of the flowfield here is x,y,dim

            if obj.dim == 3
                obj.uv = zeros(obj.l,obj.n,obj.m,obj.dim);  %3D storage order is z x y dim
            end
        end


        %Used to advance the flow field by one time step, i.e. update the wind speed field based on the colored noise.
        %Depending on dim, call step2D to update the 2D wind field; in 3D, after updating the 2D wind field on the first layer, the other height layers are scaled by the power-law exponent alph and random noise is added.
        function [obj, flag] = step(obj)

            if obj.dim == 2
                obj.uv_temp = obj.uv;
                [obj,flag] = step2D(obj);
                obj.uv = obj.uv_temp;

            elseif obj.dim == 3
                %Take the 2D velocity components u, v of layer 1; the result is one slice of the 4D array with dimensions (1, n, m, 2). squeeze removes the singleton dimension (here the 1st dimension), giving a 3D array of size n×m×2.
                %The purpose is to extract the horizontal components of the 2D velocity field of layer 1.
                obj.uv_temp = squeeze(obj.uv(1,:,:,1:2));
                [obj,flag] = step2D(obj);
                obj.uv_temp(:,:,3) = 0; %the z-direction velocity is 0
                obj.uv(1,:,:,:) = obj.uv_temp;  %velocity of the first layer; starting from the second layer, scale and add noise
                z = linspace(obj.zmin,obj.zmax,obj.l);
                %Starting from the second layer, the other height layers are scaled by the power-law exponent alph and random noise is added. As a result, the higher the altitude, the stronger the wind.
                for i = 2:obj.l
%                    obj.uv(i,:,:,:) = obj.uv_temp.*(z(i)/2).^obj.alph...
%                                      + 0.1*rand(obj.n,obj.m,3);    %n rows, m columns grid TODO: a bias was added to the wind speed here, which is very likely the cause of the data asymmetry
                    obj.uv(i,:,:,:) = obj.uv_temp.*(z(i)/2).^obj.alph...
                                      + 0.1*rand(obj.n,obj.m,3) - 0.05;    %n rows, m columns grid TODO: the offset was subtracted

                end

            else
                disp('Error: Invalid Dimension Value')
            end
        end

        %According to the passed-in sensor positions (a matrix of 2D or 3D coordinates), convert them to the corresponding discrete grid point indices in the flow field.
        %Returns the wind speed vector data at the corresponding positions.
        function ws = wind_measurement(obj,sensor)
            % sensor is a matrix
            % each row vector is (x, y, [z])    the wind speed is 3-dimensional

            L     = size(sensor, 1); % number of sensors  number of sensors (different sensors are at different positions, so the data they read should also differ)
            k = zeros(L, obj.dim);  %k is an L x dim matrix, L is the number of sensors and dim is the dimension of the wind speed

            if obj.dim == 2
                k(:,:) = round(sensor ./ ([obj.dx obj.dy])); % discrete position    divide the sensor positions element-wise by the step sizes to get the grid coordinates, completing the discretization of coordinates into grid coordinates
            elseif obj.dim == 3
                k(:,:) = round(sensor ./ ([obj.dx obj.dy obj.dz])); % discrete position divide by the step sizes of the three directions respectively
            end

            k(k(:, :) < 1) = 1; % sets lower-bound for all indices, the minimum index is 1
            k(k(:, 1) > obj.n, 1) = obj.n; % sets upper-bound maximum index in the x direction
            k(k(:, 2) > obj.m, 2) = obj.m; % sets upper-bound maximum index in the y direction

            if obj.dim == 3
                k(k(:, 3) > obj.l, 3) = obj.l; % sets upper-bound maximum index in the z direction
            end

            %The flip here is probably because obj.uv itself is stored in the order size(obj.uv) = [n_z, n_x, n_y, n_dim]
            %fliplr = flip left right, eg: k = [10 5 2], fliplr(k)-> [2 5 10]
            k = fliplr(k); % this makes it easier to index the matrix % according to the code above, for a 2D wind field, before the flip k is an (L,2) matrix

            %In MATLAB, if A is a matrix, then
            % size(A,1)   % get the number of rows of A
            % size(A,2)   % get the number of columns of A
            % size(A)     % [rows columns]

            %sub2ind function: (row, column, dimension) → linear index
            %A = rand(20,20,2);If you want A(5,10,1), you can pass the subscripts directly like this, or use sub2ind to compute the index first and then fetch
            %ind = sub2ind(size(A),5,10,1);
            %A(ind)

            % Compute the linear indices and get the corresponding wind speed components (u,v)
            % Suppose there are 3 sensors and a 2D wind field; before fliplr k is 3*2, each row being x,y, and after the flip it is y,x, still 3*2
            % Because for the 3 sensors we want the u and v wind speeds at all sensor positions (2D positions): 1*ones takes the u wind speed and 2*ones takes the v wind speed
            if obj.dim == 2
                indeces = [...
                    sub2ind(size(obj.uv), k(:, 1), k(:, 2),     ones(size(k, 1), 1)),... % u component at point x
                    sub2ind(size(obj.uv), k(:, 1), k(:, 2), 2 * ones(size(k, 1), 1)) ... % v component at point y
                    ];
            elseif obj.dim == 3
                % indeces = [...
                %     sub2ind(size(obj.uv), k(:, 1), k(:, 2),k(:, 3),     ones(size(k, 1), 1)),... % u component at point x
                %     sub2ind(size(obj.uv), k(:, 1), k(:, 2),k(:, 3), 2 * ones(size(k, 1), 1)), ... % v component at point y
                %     sub2ind(size(obj.uv), k(:, 1), k(:, 2),k(:, 3), 3 * ones(size(k, 1), 1)) ... % w component at point z
                %     ];
                %TODO: here it is 1,3,2 = z,x,y which is a bit different; not sure why, it is reversed but not completely reversed
                %obj.uv is a 4D array, with size e.g. [n_z, n_y, n_x, 3].
                 %The 4th dimension has 3 elements, representing the three velocity components:
                 %1 → u component (x-direction velocity)
                 %2 → v component (y-direction velocity)
                 %3 → w component (z-direction velocity)
                %size(k,1) is the number of points, i.e. the number of sensor points.
                %ones(size(k,1),1) generates a column vector whose length is the number of points and whose values are all 1, meaning select the 1st velocity component (u). 2* makes the values all 2, selecting the 2nd velocity component v, and *3 selects the third, w
                indeces = [...
                    sub2ind(size(obj.uv), k(:, 1), k(:, 3),k(:, 2),     ones(size(k, 1), 1)),... % u component at point x
                    sub2ind(size(obj.uv), k(:, 1), k(:, 3),k(:, 2), 2 * ones(size(k, 1), 1)), ... % v component at point y
                    sub2ind(size(obj.uv), k(:, 1), k(:, 3),k(:, 2), 3 * ones(size(k, 1), 1)) ... % w component at point z
                    ];
            end

            %After the indeces are computed, take the wind speed ws
            ws = obj.uv(indeces);
        end
    end

    %Used to get and set the current simulation time.
    methods (Static)
        function showCurrentTime(obj)
            disp(getCurrentTime(obj))
        end
        function updateCurrentTime(obj,time)
            obj = setCurrentTime(obj,time);
        end
    end

    %This function is the hardest; it does: 1. random wind boundary 2. advection diffusion PDE  3. ADI implicit solver, yielding a continuously varying 2D wind field with noise disturbance
    %here: the core 2D flow field update function, implementing the 2D flow field dynamics driven by colored noise.
    %It is complex because it is: smooth; continuous; spatially correlated; temporally correlated
    methods (Access = private)
        function [obj,flag] = step2D(obj)
            flag = 0;
            %If the colored noise buffer is empty, generate a new white noise sequence and use the filter to generate colored noise. colored noise: if there is no colored noise buffer, generate a new one
            if isempty(obj.cn) % if there are no more colored noise, generate some
                l = 10000; % arbitrary number choice % length of the generated noise
                obj.wn = randn(4, 2, l); % white noise; 2 is for 2D % generate white noise for 4 positions, 2 dimensions and length l

                % generate l sets of colored noise vectors and store it
                % the filter's state "zf" will provide memory of the previous noise, for the next time we generate more noise
                if ~isfield(obj, 'zf')
                    obj.zf = [];
%                     obj.Gzf = [];
                end
                %Gegenbauer Filter
%                 [obj.Gn, obj.Gzf] = filter(tf_Gegenbauer, 1, obj.wn, obj.Gzf, 3);
                %Colored Noise Filter use the filter to turn the white noise into colored noise, using the H(s) filter computed earlier
                [obj.cn, obj.zf] = filter(obj.tf_num, obj.tf_den, obj.wn, obj.zf, 3); % 3 refers to the third dimension being time
                flag = 1;   %flag is used to mark whether new noise was generated.
            end

            %Determine the boundary conditions through interpolation of the colored-noise "corner points" (cn_corner_vectors).
            % grab colored noise corner values
            cn_corner_vectors = obj.cn(:, :, 1); % pop one set of colored noise % take out the colored-noise vectors of the four corners as the boundary conditions, 4 corners, 2-dimensional vectors
            obj.cn(:, :, 1) = [];  % pop the noise that has been used

            % update the boundaries Take one set of noise each time to update the boundaries. Use linear interpolation to compute the velocity distribution on the four edges.
            %t,l,r,b are the velocity distributions of the top, left, right and bottom boundaries respectively, linear interpolation along the boundary, with the background wind obj.uv0.
            t = (cn_corner_vectors(1, :) - cn_corner_vectors(2, :)) ./ (1 - obj.m) .* ((1:obj.m).' - 1) + cn_corner_vectors(1, :) + obj.uv0;
            l = (cn_corner_vectors(1, :) - cn_corner_vectors(3, :)) ./ (1 - obj.n) .* ((1:obj.n).' - 1) + cn_corner_vectors(1, :) + obj.uv0;
            r = (cn_corner_vectors(2, :) - cn_corner_vectors(4, :)) ./ (1 - obj.n) .* ((1:obj.n).' - 1) + cn_corner_vectors(2, :) + obj.uv0;
            b = (cn_corner_vectors(3, :) - cn_corner_vectors(4, :)) ./ (1 - obj.m) .* ((1:obj.m).' - 1) + cn_corner_vectors(3, :) + obj.uv0;
            obj.uv_temp(1, :, :) = t;
            obj.uv_temp(:, 1, :) = l;
            obj.uv_temp(:, obj.m, :) = r;
            obj.uv_temp(obj.n, :, :) = b;

            % adding mask for buildings
            %     if isfield(ff_props, 'mask')
            %         ff = ff.*mask;
            %     end

            % first, convert n x m flowfield matrix to an nm long vector flatten into a vector
            u_n = reshape(obj.uv_temp(:, :, 1), [obj.n * obj.m 1]);
            v_n = reshape(obj.uv_temp(:, :, 2), [obj.n * obj.m 1]);

            % use this to make vectors of the same element
            e = ones(obj.n*obj.m, 1);

            % constants from Gonca ÇELIKTEN 1 and Emine Nesligul AKSAN 2 paper  these are the discrete coefficients of the advection + diffusion PDE
            rx_1 = obj.dt / 4 / obj.dx;
            ry_1 = obj.dt / 4 / obj.dy;
            rx_2 = obj.dt / 2 / obj.dx / obj.dx * obj.K(1) * 0.5;
            ry_2 = obj.dt / 2 / obj.dy / obj.dy * obj.K(2) * 0.5;

            % Use the finite-difference scheme and sparse matrices (spdiags) to construct the coefficient matrices and solve the velocity field update under the implicit difference scheme. A two-step splitting method is used to handle the x and y velocity components alternately.
            %Implicit finite-difference PDE, solving A u(k+1) = B u(k)
            % the first argument to spdiags is row vectors representing the elements along the diagonals
            % the second argument is a vector representing which diagonal the diagonals will be placed on
            % in this case, the higher indexed terms are below the main diagonal, and the lower indexed terms are above the main diagonal
            % This is because we're transposing the matrix afterwards.
            % we transpose the matrix because spdiags line up elements of the same index vertically, whereas we want them to line up horizontally

            %                      ( i-1, j term    )  ( i, j term      )   ( i+1, j term  )
            A = spdiags([-rx_1 * u_n - rx_2  (1 + 2 * rx_2) * e   rx_1 * u_n - rx_2], [obj.n 0 -obj.n], obj.n*obj.m, obj.n*obj.m).';
            B = spdiags([ ry_1 * v_n + ry_2  (1 - 2 * ry_2) * e  -ry_1 * v_n + ry_2], [obj.n 0 -obj.n], obj.n*obj.m, obj.n*obj.m).';

            % fixed boundary, the boundary is not updated; the boundary_filter function is implemented below
            % insert rows from the identity matrix the coefficient matrix to not affect boundary conditions
            A = obj.boundary_filter(A, obj.n, obj.m);
            B = obj.boundary_filter(B, obj.n, obj.m);

            %solve the PDE
            u_n5 = A\(B * u_n); % solve for u_n + 1/2
            v_n5 = A\(B * v_n); % solve for v_n + 1/2

            u_n5_l = u_n5; % linearization of u_n5
            v_n5_l = v_n5; % linearization of v_n5

            %The second splitting, alternating-direction implicit
            %                      ( i, j-1 term       )  ( i, j term      )   ( i, j+1 term     )
            A = spdiags([-ry_1 * v_n5_l - ry_2  (1 + 2 * ry_2) * e   ry_1 * v_n5_l - ry_2], [1 0 -1], obj.n*obj.m, obj.n*obj.m).';
            B = spdiags([ rx_1 * u_n5_l + rx_2  (1 - 2 * rx_2) * e  -rx_1 * u_n5_l + rx_2], [1 0 -1], obj.n*obj.m, obj.n*obj.m).';

            % put identity into the matrix to not affect boundary conditions
            % Use boundary_filter to keep the boundary conditions unchanged (the boundary rows are replaced by identity rows to prevent the boundary from changing).
            A = obj.boundary_filter(A, obj.n, obj.m);
            B = obj.boundary_filter(B, obj.n, obj.m);

            %solve again to get the wind field at the next time step
            u_n1 = A\(B * u_n5);
            v_n1 = A\(B * v_n5);

            % reshape back to the grid, restoring the n * m * 2 wind field
            % convert nm long vector back into n x m flowfield matrix
            obj.uv_temp(:, :, 1) = reshape(u_n1, [obj.n obj.m]);
            obj.uv_temp(:, :, 2) = reshape(v_n1, [obj.n obj.m]);

%             % apply mask again
%             if isfield(obj, 'mask')
%                 mask   = obj.mask;
%                 obj.uv_temp = obj.uv_temp.*mask;
%             end

        end

        %TODO: don't understand
        %Used to process the coefficient matrix, ensuring that the linear equations corresponding to the boundary positions are identity equations so that the boundary values remain unchanged.
        %By replacing the boundary rows with identity-matrix rows, a "fixed boundary" is implemented. That is u_new = u_old; the boundary is fixed and does not take part in the PDE update
        function [A] = boundary_filter(obj,A, n, m)
            % Puts identity rows into the A matrix as to leave the boundaries alone
            I = eye(n*m, 'like', A);
            % left boundary
            A(1:n, :) = I(1:n, :);
            % right boundary
            A(n*(m-1):n*m, :) = I(n*(m-1):n*m, :);
            % top boundary
            A(1:n:n*m, :) = I(1:n:n*m, :);
            % bottom boundary
            A(n:n:n*m, :) = I(n:n:n*m, :);
        end

        function [time]=getCurrentTime(obj)
            time = obj.currentTime;
        end
        function [obj] = setCurrentTime(obj,time)
            obj.currentTime = time;
        end
    end
    
end