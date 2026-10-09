save_folder = 'generated_data/G=1_80_60_modified_Q_true_V7_eta_SNR_10_10000s_wind_and_quant_check';
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

n_rows = size(DATA_Height_5, 1);   % total number of rows, i.e. 100*n
chunk_size = 100 * 1000;           % number of rows per chunk (corresponds to 1000 time steps at 100 rows each); adjust according to available memory

c_filepath = fullfile(save_folder, 'Plume-C-Data-Height-5.mat');
ws_filepath = fullfile(save_folder, 'Plume-Wind-Data-Height-5.mat');

% Use matfile to create files in v7.3 format that can be written chunk by
% chunk, so there is no need to hold the whole C/ws arrays in memory at once
mc = matfile(c_filepath, 'Writable', true);
mw = matfile(ws_filepath, 'Writable', true);

n_cols_c = size(DATA_Height_5, 2) / 4 * 1;   % C takes 1/4 of the columns (1 out of every 4 columns)
n_cols_ws = size(DATA_Height_5, 2) / 4 * 3;  % ws takes the remaining 3/4 of the columns

% Key addition: first use this line as a "placeholder" to tell matfile the final shape of the whole variable, writing only the last element, without having to hold the full array in memory; only then can the loop use colon indexing for partial assignment
mc.C(n_rows, n_cols_c) = 0;
mw.ws(n_rows, n_cols_ws) = 0;

tic;  % New: record the save start time
n_chunks = ceil(n_rows / chunk_size);
chunk_idx = 0;

for start_row = 1:chunk_size:n_rows
    chunk_idx = chunk_idx + 1;   % <-- New: add this line; missing it was the root of the problem
    end_row = min(start_row + chunk_size - 1, n_rows);
    block = DATA_Height_5(start_row:end_row, :);   % take only a small block, not everything

    c_block = block(:, 4:4:end);
    ws_block = block;
    ws_block(:, 4:4:end) = [];

    mc.C(start_row:end_row, :) = c_block;
    mw.ws(start_row:end_row, :) = ws_block;

    % New: print the elapsed time and estimated remaining time every time (the
    % number of chunks is small, so unlike the main loop there is no need to
    % print only once every few hundred steps; printing for every chunk will not flood the screen)
    elapsed_sec = toc;
    avg_sec_per_chunk = elapsed_sec / chunk_idx;
    est_remaining_sec = avg_sec_per_chunk * (n_chunks - chunk_idx);
    fprintf('Saved %d / %d rows (chunk %d/%d), elapsed=%.1f min, est_remaining=%.1f min\n', ...
        end_row, n_rows, chunk_idx, n_chunks, elapsed_sec/60, est_remaining_sec/60);
end

disp('Chunked save finished');