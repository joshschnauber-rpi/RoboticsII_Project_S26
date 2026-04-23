clear; clc; close all;
rng('shuffle');



%% Video saving parameters
save_video = false;
video_filename = 'vid_1.mp4';
video_fps = 15;


%% Simulation parameters
init_robot_conf = [0, 0, 0];
goal_pos = [27, 27];
goal_distance_tol = 1.0;
world = [-5 30; -5 30];   % boundary for path traversion (changeable)

L = 1.0;
K_v = 5.0;
K_h = 1.0;
max_speed = 2.0;
max_gamma = pi / 6;
dt = 0.1;
maxSteps = 1000;

obs_distance_factor = 30.0;

% Measurement noise distribution
Q_robot_conf = diag([0.12, 0.12, 0.01]);
Q_obs_center = diag([0.15, 0.15]);
Q_obs_vel = diag([0.15, 0.15]);

% Movement noise
R_robot = diag([0.005, 0.005, 0.002]); % 

% Initialize robot initial pose/state
robot_conf = init_robot_conf;

% Initialize moving obstacles
num_obs = 3;
obs.radii = 2.5 + 1.5*rand(1, num_obs);   % randomizing the radius between 2.5 and 4.0
obs.centers = zeros(num_obs, 2);
obs.vels = zeros(num_obs, 2);

for j = 1:num_obs
    valid = false;
    while ~valid
        % ensure obstacles bounce from their boundary
        cx = world(1, 1) + obs.radii(j) + (world(1, 2)-world(1, 1)-2*obs.radii(j))*rand;
        cy = world(2, 1) + obs.radii(j) + (world(2, 2)-world(2, 1)-2*obs.radii(j))*rand;

        ctest = [cx; cy];

        % obstacles not starting near the start or goal position
        far_from_start = norm(ctest - robot_conf(1:2)) > 8;
        far_from_goal  = norm(ctest - goal_pos) > 8;

        % stop obstacles from overlapping initially
        no_overlap = true;
        for m = 1:j-1
            if norm(ctest - obs.centers(m,:)) < (obs.radii(j) + obs.radii(m) + 3)
                no_overlap = false;
                break;
            end
        end

        valid = far_from_start && far_from_goal && no_overlap;
    end

    obs.centers(j,:) = ctest;

    % random straight line speed
    speed = 0.25 + (2 - 0.25)*rand;   % speed between 0.25 and 2
    ang = 2*pi*rand;                  % random line/vector for direction
    obs.vels(j,:) = speed*[cos(ang); sin(ang)];
end

% Initialize estimation
[robot_conf_z, obs_vel_z, obs_centers_z] = measure_environment(robot_conf, obs.vels, obs.centers, Q_robot_conf, Q_obs_vel, Q_obs_center);
robot_conf_est = robot_conf_z;
robot_conf_cov = Q_robot_conf;
u = [0 0 0];

% Particle creation
num_particles = 20^2;
xp = linspace(world(1, 1), world(1, 2), sqrt(num_particles));
yp = linspace(world(2, 1), world(2, 2), sqrt(num_particles));
[X, Y] = meshgrid(xp, yp);
particle_pos = [X(:), Y(:)];
particle_weights = zeros(num_particles, 1);

particle_distance = 2.0 * (world(1, 2) - world(1, 1)) / sqrt(num_particles);
max_distance = 1.5*norm(world(:,1) - world(:,2));

particle_neighbors = cell(num_particles,1);
for i = 1:num_particles
    particle_neighbors{i} = find_neighbors(particle_pos, particle_pos(i,:), particle_distance);
end


%% Storing/initializing data points
robot_conf_hist(1,:) = robot_conf;
robot_conf_est_hist(1,:) = robot_conf;

obs_centers_hist = cell(1, num_obs);
for j = 1:num_obs
    obs_centers_hist{j} = obs.centers(j,:);
end

%% Figures
fig1 = figure('Color','r','Position',[100 100 1000 750]);
fig1.Color = 'w';
set(gca, 'Color', 'w');
hold on; grid on; axis equal;
xlim(world(1,:));
ylim(world(2,:));
xlabel('x position', 'Color', 'k');
ylabel('y position', 'Color', 'k');
title('Particle Field with Moving Obstacles - Real-Time Trajectory', 'Color', 'k');

h_robot_conf_z =    plot(robot_conf(1), robot_conf(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Measured Position');
h_robot_conf_est =  plot(robot_conf(1), robot_conf(2), 'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 8, 'DisplayName', 'Estimated Position');
h_goal =            plot(goal_pos(1), goal_pos(2), 'g*', 'MarkerSize', 14, 'DisplayName', 'Goal');
h_robot_path =      plot(robot_conf_hist(:,1), robot_conf_hist(:,1), 'k-', 'LineWidth', 2, 'DisplayName', 'True Path');
h_robot_est_path =  plot(robot_conf_est_hist(:,1), robot_conf_est_hist(:,2), 'm--', 'LineWidth', 2, 'DisplayName', 'Estimated Path');
h_robot_target_path=plot(robot_conf_hist(:,1), robot_conf_hist(:,2), 'b--', 'LineWidth', 2, 'DisplayName', 'Target Path');
h_particles =       scatter(particle_pos(:,1), particle_pos(:,2), 16, particle_weights, 'filled', 'DisplayName', 'Particles');
clim([0, 15]);
h_robot_conf =      DrawRobot(1,2,robot_conf(1),robot_conf(2),robot_conf(3));
h_obs_circles =     gobjects(1, num_obs);
h_obs_centers_est = gobjects(1, num_obs);
h_obs_paths =       gobjects(1, num_obs);
for j = 1:num_obs
    [xc, yc] = circle_points(obs.centers(j,:), obs.radii(j));
    h_obs_circles(j) =      plot(xc, yc, 'r-', 'LineWidth', 2);
    h_obs_centers_est(j) =  plot(obs.centers(j,1), obs.centers(j,2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    h_obs_paths(j) =        plot(obs_centers_hist{j}(:,1), obs_centers_hist{j}(:,2), 'Color', 'black', ...
                                'LineStyle', '--', 'LineWidth', 1.5);
end

lgd = legend({'Measured Position', 'Estimated Position', 'Goal', 'True Path', 'Estimated Path', 'Target Path'}, 'Location', 'bestoutside', 'Color', 'k');
lgd.Color = 'w';
colorbar;

if save_video
    vobj = VideoWriter(video_filename, 'MPEG-4');
    vobj.FrameRate = video_fps;
    vobj.Quality = 100;
    open(vobj);

    drawnow;
    frame = getframe(fig1);
    writeVideo(vobj, frame);
end



%% Control and movement action simulation
for k = 1:maxSteps
    % Terminate near goal
    if norm(robot_conf(1:2) - goal_pos) < goal_distance_tol
        break;
    end

    % Sense things in environment
    [robot_conf_z, obs_vels_z, obs_centers_z] = measure_environment(robot_conf, obs.vels, obs.centers, Q_robot_conf, Q_obs_vel, Q_obs_center);

    % Estimate robot state
    robot_conf_cov_bar = robot_conf_cov + R_robot;
    robot_conf_est_bar = robot_conf_est + (dt * u);

    K = robot_conf_cov_bar / (robot_conf_cov_bar + Q_robot_conf);
    robot_conf_cov = (eye(3) - K) * robot_conf_cov_bar;
    robot_conf_est = robot_conf_est_bar + transpose(K * transpose(robot_conf_z - robot_conf_est_bar)); 

    % 'Estimate' obstacle state
    obs_vels_est = obs_vels_z;
    obs_centers_est = obs_centers_z;


    % Weigh each particle based on proximity to goal and obstacles
    for i = 1:num_particles
        pos = particle_pos(i,:);
        
        % Weight higher based on further distance from obstacles
        obs_w = 0;
        for j = 1:num_obs
            obs_pos = obs_centers_est(j,:);
            obs_radius = obs.radii(j);
            obs_dist = max(norm(pos - obs_pos) - obs_radius, eps);
            obs_w = obs_w + (obs_distance_factor / obs_dist);
        end
        
        particle_weights(i) = obs_w;
    end

    % Use A* to find path
    target_path = A_star(particle_pos, particle_weights, particle_neighbors, robot_conf_est(1:2), goal_pos, particle_distance);
    if isempty(target_path)
        target_pos = robot_conf_est(1:2);
        target_path(end+1,:) = target_pos;
        disp("No path found");
    else
        target_pos = target_path(2,:);
    end
    
    % Determine robot movement to move towards target
    err = K_v * norm(target_pos - robot_conf_est(1:2));
    speed = min(err, max_speed);
    target_theta = atan2(target_pos(2) - robot_conf_est(2), target_pos(1) - robot_conf_est(1));
    gamma = K_h * wrapToPi_local(target_theta - robot_conf(3));
    gamma = clip(gamma, -max_gamma, max_gamma);
    

    % Store data points
    robot_conf_hist(end+1,:) = robot_conf;
    robot_conf_est_hist(end+1,:) = robot_conf_est;
    for j = 1:num_obs
        obs_centers_hist{j}(end+1,:) = obs.centers(j,:);
    end

    % Live plot updating    
    delete(h_robot_conf);
    h_robot_conf =          DrawRobot(1,2,robot_conf(1),robot_conf(2),robot_conf(3));
    set(h_robot_conf_z,     'XData', robot_conf_z(1),           'YData', robot_conf_z(2));
    set(h_robot_conf_est,   'XData', robot_conf_est(1),         'YData', robot_conf_est(2));
    set(h_robot_path,       'XData', robot_conf_hist(:,1),      'YData', robot_conf_hist(:,2));
    set(h_robot_est_path,   'XData', robot_conf_est_hist(:,1),  'YData', robot_conf_est_hist(:,2));
    set(h_robot_target_path,'XData', target_path(:,1),          'YData', target_path(:,2));
    set(h_particles,        'XData', particle_pos(:,1),         'YData', particle_pos(:,2), 'CData', particle_weights);
    for j = 1:num_obs
        [xc, yc] = circle_points(obs.centers(j,:), obs.radii(j));
        set(h_obs_circles(j),       'XData', xc,                        'YData', yc);
        set(h_obs_centers_est(j),   'XData', obs_centers_est(j,1),      'YData', obs_centers_est(j,2));
        set(h_obs_paths(j),         'XData', obs_centers_hist{j}(:,1),  'YData', obs_centers_hist{j}(:,2));
    end
    drawnow;

    % Update video
    if save_video
        frame = getframe(fig1);
        writeVideo(vobj, frame);
    end


    % Update robot state
    u = motion_model(robot_conf, speed, gamma, L, dt);
    robot_conf = robot_conf + u + sample_normal(zeros(1, 3), R_robot);
    robot_conf(3) = wrapToPi_local(robot_conf(3));

    % Update obstacle states
    obs = update_obstacles(obs, dt, world);

    
    pause(dt * 0.1);
end



%% Display results
disp(['Goal reached in ', num2str(k), ' steps.']);
disp(['Elapsed time = ', num2str(k*dt), ' seconds.']);

%% Video closing
if save_video
    close(vobj);
    disp(['Video saved as: ', video_filename]);
end



%% Helper Functions
function a = wrapToPi_local(a)
    a = mod(a + pi, 2*pi) - pi;
end

function s = sample_normal(mean, cov)
    dim = size(mean);
    s = mean + mvnrnd(zeros(dim(2), 1), cov, dim(1));
end

function [robot_conf_z, obs_vels_z, obs_centers_z] = measure_environment(robot_conf, obs_vel, obs_centers, Q_robot_conf, Q_obs_vel, Q_obs_center)
    % Get robot state from sensor
    robot_conf_z = sample_normal(robot_conf, Q_robot_conf);
    robot_conf_z(3) = wrapToPi_local(robot_conf_z(3));
    % Get obstacle state from sensor
    obs_vels_z = sample_normal(obs_vel, Q_obs_vel);
    obs_centers_z = sample_normal(obs_centers, Q_obs_center);
end

% Find neighbors of particle
function neighbors = find_neighbors(particles, particle, neighbor_dist)    
    dists = vecnorm(particles - particle, 2, 2);
    neighbors = transpose(find(dists <= neighbor_dist & dists > 0));
end

% Reconstruct path backwards
function path = reconstruct_path(came_from, current, points)
    path = points(current,:);
    while came_from(current) ~= 0
        current = came_from(current);
        path = [points(current,:); path];
    end
end

function path = A_star(particles, particle_weights, particle_neighbors, start_pos, goal_pos, neighbor_dist)
    % Add start and end points to particles
    particles(end+1,:) = goal_pos;
    particle_weights(end+1) = 0;
    particle_neighbors{end+1} = find_neighbors(particles, goal_pos, neighbor_dist);
    particles(end+1,:) = start_pos;
    particle_weights(end+1) = 0;
    particle_neighbors{end+1} = find_neighbors(particles, start_pos, neighbor_dist);
    
    num_particles = size(particles, 1);
    start_i = num_particles;
    goal_i  = num_particles-1;

    for i = 1:num_particles
        pos = particles(i,:);
        if norm(goal_pos - pos) < neighbor_dist
            particle_neighbors{i}(end+1) = goal_i;
        end
    end

    % A* initialization
    g = inf(num_particles,1);
    f = inf(num_particles,1);
    function h = h(particle_i)
        h = norm(particles(particle_i,:) - goal_pos);
    end

    g(start_i) = 0;
    f(start_i) = h(start_i);

    open_set = start_i;
    came_from = zeros(num_particles,1);

    while ~isempty(open_set)
        % Get node in open_set with lowest f
        [~, min_set_i] = min(f(open_set));
        current_i = open_set(min_set_i);

        if current_i == goal_i
            path = reconstruct_path(came_from, current_i, particles);
            return;
        end

        open_set(min_set_i) = [];
        current_particle = particles(current_i,:);

        % Explore neighbors
        for neighbor_i = particle_neighbors{current_i}
            neighbor_particle = particles(neighbor_i,:);

            dist = norm(neighbor_particle - current_particle) + particle_weights(neighbor_i);
            potential_g = g(current_i) + dist;

            if potential_g < g(neighbor_i)
                came_from(neighbor_i) = current_i;
                g(neighbor_i) = potential_g;
                f(neighbor_i) = potential_g + h(neighbor_i);

                if ~ismember(neighbor_i, open_set)
                    open_set(end+1) = neighbor_i;
                end
            end
        end
    end

    path = [];
end

function u = motion_model(x, v, gamma, L, dt)
    theta = x(3);
    u = [v*cos(theta)*dt, v*sin(theta)*dt, (v/L)*tan(gamma)*dt];
end

function obs = update_obstacles(obs, dt, world)
    % Move each circular obstacle, bounce off boundaries,
    % and prevent overlap with other obstacles.

    xmin = world(1, 1); xmax = world(1, 2);
    ymin = world(2, 1); ymax = world(2, 2);

    num_obs = length(obs.radii);

    % move all obstacles
    for j = 1:num_obs
        obs.centers(j,:) = obs.centers(j,:) + obs.vels(j,:) * dt;
    end

    % bounce off walls
    for j = 1:num_obs
        r = obs.radii(j);

        % x direction
        if obs.centers(j,1) - r < xmin
            obs.centers(j,1) = xmin + r;
            obs.vels(j,1) = -obs.vels(j,1);
        elseif obs.centers(j,1) + r > xmax
            obs.centers(j,1) = xmax - r;
            obs.vels(j,1) = -obs.vels(j,1);
        end

        % y direction
        if obs.centers(j,2) - r < ymin
            obs.centers(j,2) = ymin + r;
            obs.vels(j,2) = -obs.vels(j,2);
        elseif obs.centers(j,2) + r > ymax
            obs.centers(j,2) = ymax - r;
            obs.vels(j,2) = -obs.vels(j,2);
        end
    end

    % prevent obstacle-obstacle overlap
    buffer = 0.3;

    for i = 1:num_obs-1
        for j = i+1:num_obs
            c1 = obs.centers(i,:);
            c2 = obs.centers(j,:);

            dvec = c2 - c1;
            dist = norm(dvec);

            minDist = obs.radii(i) + obs.radii(j) + buffer;

            if dist < minDist
                if dist < 1e-6
                    dir = [1; 0];
                else
                    dir = dvec / dist;
                end

                overlap = minDist - dist;
                obs.centers(i,:) = obs.centers(i,:) - 0.5 * overlap * dir;
                obs.centers(j,:) = obs.centers(j,:) + 0.5 * overlap * dir;

                % collision reaction
                obs.vels(i,:) = -obs.vels(i,:);
                obs.vels(j,:) = -obs.vels(j,:);
            end
        end
    end
end

function [xc, yc] = circle_points(center, radius)
    ang = linspace(0, 2*pi, 200);
    xc = center(1) + radius*cos(ang);
    yc = center(2) + radius*sin(ang);
end

function plot_fig = DrawRobot( width, height, center_x, center_y, theta)
        
    corner1 = [center_x - 0.25*height * cos(theta) + (width/2) * sin(theta), center_y - (0.25*height)*sin(theta) - (width/2)*cos(theta)];
    corner2 = [center_x - 0.25*height*cos(theta) - width/2*sin(theta),center_y - 0.25*height*sin(theta) + width/2*cos(theta)];
    corner3 = [center_x + 0.75*height*cos(theta) - width/2*sin(theta),center_y + 0.75*height*sin(theta) + width/2*cos(theta)];
    corner4 = [center_x + 0.75*height*cos(theta) + width/2*sin(theta),center_y + 0.75*height*sin(theta) - width/2*cos(theta)];

    corners = [corner1;corner2;corner3;corner4;corner1];
    corners = transpose(corners);
    x = [center_x, center_y];
    y = [center_x+height*cos(theta), center_y+height*sin(theta)];
    plot_fig(1) = plot([x(1), y(1)], [x(2), y(2)], 'k-', 'HandleVisibility', 'off'); hold on;

    plot_fig(2) = plot(corners(1,:),corners(2,:),'b', 'HandleVisibility', 'off');
        
end