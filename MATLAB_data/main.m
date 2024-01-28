%% Initialization
clear; close; clc;

%% (1A) Plot epoch vs. data

data = load( 'data_square.mat' );

x = data.epochs;
y = 100 * data.acc_arr;

y_mean = mean( y, 2 );
y_std  = std( y' );

f = figure( ); a = axes( 'parent', f );
hold on
scatter( x, y, 200, 'o', 'filled', 'markeredgecolor', 'k', 'markerfacecolor', 'w' , 'linewidth', 5)
a.XScale = 'log';

% errorbar( x, y_mean, y_std, 'linewidth',5 )


xlabel( 'Epochs (-)', 'fontsize', 40 );
ylabel( 'Accuracy (\%)', 'fontsize', 40 );
set( a, 'fontsize', 40, 'xlim', [ 0.1, max( x )] )

%% (1B) Dataset to sample

data = load( 'data_square_to_sample.mat' );

% Get min and max value
% Note that there are points that are out of the scope of [0,1]
% Ignore that

val_min = [ 50, 50, 50, 5, 5, 5, 0.1, 0.1 ];
val_max = [ 1000, 1000, 1000, 200, 200, 200, 1.0, 1.0 ];


N = length( data.outputs );
recovered = zeros( N, 8 );

for i = 1 : N
    tmp = data.sampled_vals( i, : );
    recovered( i, : ) = val_min + ( val_max - val_min ) .* tmp;
    
end

A = recovered;

% Plot the values on 
% Calculate mean and standard deviation
means = mean(A);
std_devs = std(A);

% Define x-axis
x = 1:8;

% Create a scatter plot for each dimension
f = figure; a = axes( 'parent', f );
hold on; % Hold on to the current figure

% Plot each sample point
for i = 1:8
    scatter( a, repmat(i, size(A, 1), 1), A(:, i), 'b.'); % Scatter plot for each dimension
end

% Errorbar for mean and standard deviation
errorbar( a, x, means, std_devs, 'r', 'LineStyle', 'none', 'LineWidth', 3);

% Add labels and legend
xlabel('Dimension');
ylabel('Values');
set( a,'xtick', 1:8, 'xticklabel', { '$k_x$', '$k_y$', '$k_z$', '$k_A$', '$k_B$', '$k_C$', '$\gamma_t$', '$\gamma_r$' }, 'fontsize', 40 )

% Scatter for min and max values
scatter(a, x, val_min, 500, 'g', 'filled', 'DisplayName', 'Min Values');
scatter(a, x, val_max, 500, 'm', 'filled', 'DisplayName', 'Max Values');

title('Scatter Plot with Mean and Standard Deviation');
grid on; % Add grid for better readability
hold off; % Release the figure
