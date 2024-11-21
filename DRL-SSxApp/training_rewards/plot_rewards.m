% MATLAB Script to Plot Rewards over Timesteps for a DRL Agent
% Author: Joshua Moore
% Date: 11/20/24

function plot_rewards(filename)
    % Validate input arguments
    if nargin < 1
        error('Please provide a filename as an argument.');
    end

    % Check if the file exists
    if ~isfile(filename)
        error('File "%s" does not exist.', filename);
    end

    % Read the CSV file
    data = readtable(filename);

    % Check if the file contains the 'Rewards' column
    if ~ismember('Rewards', data.Properties.VariableNames)
        error('File "%s" does not contain a "Rewards" column.', filename);
    end

    % Extract the rewards column
    rewards = data.Rewards;
    numTimesteps = length(rewards);  % Get the total number of timesteps

    % Generate timestep numbers (1, 2, ..., N)
    timestepNumbers = 1:numTimesteps;

    % Plot rewards over timesteps
    figure;
    plot(timestepNumbers, rewards, 'b-', 'LineWidth', 1.5);
    xlabel('Timesteps');
    ylabel('Rewards');
    title(sprintf('Training Rewards over Timesteps: %s', filename), 'Interpreter', 'none');
    grid on;

    % Adjust plot aesthetics
    set(gca, 'FontSize', 12);
end
