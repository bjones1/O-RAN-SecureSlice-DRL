function DCF(filename)
    % This function loads data from the given CSV file, processes it, and plots the DCF plot.

    % Ensure the table uses original column names by preserving the original headers
    opts = detectImportOptions(filename, 'VariableNamingRule', 'preserve');
    data = readtable(filename, opts);

    % Separate the data by Model type
    models = unique(data.Model);

    % Set up the figure for plotting
    figure;
    hold on;

    % Colors for different models
    colors = lines(length(models));

    % Loop through each model and plot its data
    for i = 1:length(models)
        model_data = data(strcmp(data.Model, models{i}), :);
        % Access columns by the original header names using dynamic field names
        plot(model_data.("Malicious Chance"), model_data.Result, '-o', ...
            'DisplayName', models{i}, 'Color', colors(i,:));
    end

    % Adding labels and title
    xlabel('Malicious Chance');
    ylabel('Result');
    title('DCF Plot for Different Models');
    legend('show');

    % Set the grid for better visualization
    grid on;
    hold off;
end
