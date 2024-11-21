function plot_all_rewards()
    % Get a list of all CSV files in the current directory
    csvFiles = dir('*.csv');

    % Check if any CSV files are found
    if isempty(csvFiles)
        error('No CSV files found in the current directory.');
    end

    % Initialize a figure
    figure;
    hold on;

    % Loop through each file and plot the average rewards
    for i = 1:length(csvFiles)
        % Read the CSV file
        data = readtable(csvFiles(i).name);

        % Check if the file contains the 'Reward' column
        if ~ismember('Reward', data.Properties.VariableNames)
            warning('File %s does not contain a "Reward" column. Skipping.', csvFiles(i).name);
            continue;
        end

        % Extract the reward column
        rewards = data.Reward;
        numEpisodes = length(rewards);
        fprintf('File: %s, Total Episodes: %d\n', csvFiles(i).name, numEpisodes);

        % Initialize arrays for averaged rewards and episode numbers
        avgRewards = [];
        episodeNumbers = [];

        % Loop through the rewards in chunks of 1000
        for j = 1:1000:numEpisodes
            % Define the range for the current chunk (up to 1000 episodes)
            endIdx = min(j + 999, numEpisodes);
            
            % Average the rewards in this chunk
            avgRewards = [avgRewards; mean(rewards(j:endIdx))];

            % The x-axis value will be the last episode of the chunk
            episodeNumbers = [episodeNumbers; endIdx];
        end

        % Adjust the x-axis to show actual episodes
        % We already have the correct episode number in episodeNumbers
        % The x-axis will show cumulative episodes up to each point

        % Determine the label based on filename (e.g., 'DQN', 'DDQN', 'Dueling DQN')
        if contains(csvFiles(i).name, 'DQN', 'IgnoreCase', true)
            if contains(csvFiles(i).name, 'DDQN', 'IgnoreCase', true)
                label = 'DDQN';
            elseif contains(csvFiles(i).name, 'Dueling_DQN', 'IgnoreCase', true)
                label = 'Dueling DQN';
            else
                label = 'DQN';
            end
        else
            label = csvFiles(i).name; % Default to the filename if no keywords are found
        end

        % Plot the averaged rewards with correct episode numbers on x-axis
        plot(episodeNumbers, avgRewards, 'LineWidth', 1.5, 'DisplayName', label);
    end

    % Add labels, legend, and title
    xlabel('Timesteps');
    ylabel('Average Reward');
    title('Average Reward over Episodes for All Files');
    legend('show', 'Location', 'best');
    grid on;

    % Adjust plot aesthetics
    set(gca, 'FontSize', 12);

    hold off;
end
