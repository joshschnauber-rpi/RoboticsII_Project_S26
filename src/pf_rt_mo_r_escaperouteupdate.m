clear; clc; close all;
rng('shuffle');

%% Parameters
dt = 0.1;
L = 1.0;
N = 500;
maxSteps = 600;

goal = [27; 27];
world = [-5 30 -5 30];   % boundary for path traversion (changeable)

% Measurement noise distribution
R = diag([0.01, 0.01, 0.002]);
Q = diag([0.12, 0.12, 0.01]);
R_sqrt = chol(R, 'lower');
Q_sqrt = chol(Q, 'lower');

% Obstacle noise (measured)
obs_sigma = 0.45;      % noise center measurements
vel_sigma = 0.08;      % noise velocity measurements
pred_horizon = 1.0;    % "prediction" of movement factor

% Video saving stuff
save_video = true;
video_filename = 'pf_moving_obstacles_escape_mode_3.mp4';
video_fps = 15;

% robot initial pose/state
x_true = [0; 0; 0];

% Initial robot measurement
z0 = x_true + Q_sqrt * randn(3,1);
z0(3) = wrapToPi_local(z0(3));

% particle creation
particles = zeros(3,N);
particles(1,:) = z0(1) + 2.0*randn(1,N);
particles(2,:) = z0(2) + 2.0*randn(1,N);
particles(3,:) = wrapToPi_local(z0(3) + 0.25*randn(1,N));
weights = ones(1,N) / N;
x_est = particle_mean(particles, weights);

%% Moving obstacles
numObs = 3;

obs.radii = 2.5 + 1.5*rand(1,numObs);   % randomizing the radius between 2.5 and 4.0
obs.centers = zeros(2,numObs);
obs.vel = zeros(2,numObs);

for j = 1:numObs
    valid = false;
    while ~valid
        % ensure obstacles bounce from their boundary
        cx = world(1) + obs.radii(j) + (world(2)-world(1)-2*obs.radii(j))*rand;
        cy = world(3) + obs.radii(j) + (world(4)-world(3)-2*obs.radii(j))*rand;

        ctest = [cx; cy];

        % obstacles not starting near the start or goal position
        far_from_start = norm(ctest - x_true(1:2)) > 8;
        far_from_goal  = norm(ctest - goal) > 8;

        % stop obstacles from overlapping initially
        no_overlap = true;
        for m = 1:j-1
            if norm(ctest - obs.centers(:,m)) < (obs.radii(j) + obs.radii(m) + 3)
                no_overlap = false;
                break;
            end
        end

        valid = far_from_start && far_from_goal && no_overlap;
    end

    obs.centers(:,j) = ctest;

    % random straight line speed
    speed = 0.25 + (2 - 0.25)*rand;   % speed between 0.25 and 2
    ang = 2*pi*rand;                  % random line/vector for direction
    obs.vel(:,j) = speed*[cos(ang); sin(ang)];
end

% initializing obstacle measurements with noise
obs_meas = obs.centers + obs_sigma * randn(size(obs.centers));
obs_vel_meas = obs.vel + vel_sigma * randn(size(obs.vel));

%% storing/initializing data points
true_hist = x_true;
est_hist  = x_est;
meas_hist = z0;
Neff_hist = [];

obs_hist = cell(1, numObs);
for j = 1:numObs
    obs_hist{j} = obs.centers(:,j);
end

%% Stuck detection / escape mode settings
goal_dist_hist = norm(x_est(1:2) - goal);
stuck_count = 0;
escape_mode = false;
escape_timer = 0;
escape_dir = 1;              %d determine which way to turn

progress_window = 12;        % history of steps 
progress_thresh = 0.35;      % goal distance comparative
stuck_trigger = 6;           % how many progress checks before turning
escape_duration = 18;        % turn away will last 18 steps unless path is clear earlier
escape_clearance_thresh = 4.0;

%% Figures
fig1 = figure('Color','w','Position',[100 100 1000 750]);
hold on; grid on; axis equal;
xlim(world(1:2));
ylim(world(3:4));
xlabel('x position');
ylabel('y position');
title('Particle Filter with Moving Obstacles - Real-Time Trajectory');

hParticles = scatter(particles(1,:), particles(2,:), 8, 'b', 'filled', 'DisplayName', 'Particles');

hTruePath = plot(true_hist(1,:), true_hist(2,:), 'k-', 'LineWidth', 2, 'DisplayName', 'True Path');
hEstPath = plot(est_hist(1,:), est_hist(2,:), 'm--', 'LineWidth', 2, 'DisplayName', 'Estimated Path');

hTrueNow = plot(x_true(1), x_true(2), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 8, 'DisplayName', 'True Position');
hEstNow = plot(x_est(1), x_est(2), 'mo', 'MarkerFaceColor', 'm', 'MarkerSize', 8, 'DisplayName', 'Estimated Position');
hMeasNow = plot(z0(1), z0(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Robot Measurement');

hGoal = plot(goal(1), goal(2), 'g*', 'MarkerSize', 14, 'DisplayName', 'Goal');

hObsCircle = gobjects(1, numObs);
hObsTrail  = gobjects(1, numObs);
hObsMeas   = gobjects(1, numObs);

for j = 1:numObs
    [xc, yc] = circle_points(obs.centers(:,j), obs.radii(j));
    hObsCircle(j) = plot(xc, yc, 'k-', 'LineWidth', 2);
    hObsTrail(j)  = plot(obs_hist{j}(1,:), obs_hist{j}(2,:), 'Color', [0.2 0.6 0.2], ...
                         'LineStyle', '--', 'LineWidth', 1.5);
    hObsMeas(j)   = plot(obs_meas(1,j), obs_meas(2,j), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
end

legend({'Particles','True Path','Estimated Path','True Position','Estimated Position','Robot Measurement','Goal'}, ...
    'Location','bestoutside');

% video saving/opening
if save_video
    vobj = VideoWriter(video_filename, 'MPEG-4');
    vobj.FrameRate = video_fps;
    vobj.Quality = 100;
    open(vobj);

    drawnow;
    frame = getframe(fig1);
    writeVideo(vobj, frame);
end

%% Control and movement action
invQ = inv(Q);

for k = 1:maxSteps

    % update obstacle motion
    obs = update_obstacles(obs, dt, world);

    % updating obstacle measurements with noise
    obs_meas = obs.centers + obs_sigma * randn(size(obs.centers));
    obs_vel_meas = obs.vel + vel_sigma * randn(size(obs.vel));

    % checking if trapped
    goal_dist = norm(x_est(1:2) - goal);
    goal_dist_hist(end+1) = goal_dist;

    if length(goal_dist_hist) > progress_window
        progress = goal_dist_hist(end-progress_window) - goal_dist_hist(end);

        if progress < progress_thresh && goal_dist > 3.0
            stuck_count = stuck_count + 1;
        else
            stuck_count = max(stuck_count - 1, 0);
        end
    end

    if ~escape_mode && stuck_count >= stuck_trigger
        escape_mode = true;
        escape_timer = escape_duration;
        escape_dir = choose_escape_direction(x_est, goal, obs_meas);
    end

    % control (robot and obstacle measurement
    [v, gamma, min_clearance] = controller_with_moving_obstacles( ...
        x_est, goal, obs_meas, obs_vel_meas, obs.radii, pred_horizon, ...
        escape_mode, escape_dir);

    % update escape state
    if escape_mode
        escape_timer = escape_timer - 1;

        if min_clearance > escape_clearance_thresh || escape_timer <= 0
            escape_mode = false;
            stuck_count = 0;
        end
    end

    % path control
    x_true = motion_model(x_true, v, gamma, L, dt);
    x_true = x_true + R_sqrt * randn(3,1);
    x_true(3) = wrapToPi_local(x_true(3));

    % robot pose update
    z = x_true + Q_sqrt * randn(3,1);
    z(3) = wrapToPi_local(z(3));

    % particle propagation
    for i = 1:N
        particles(:,i) = motion_model(particles(:,i), v, gamma, L, dt);
        particles(:,i) = particles(:,i) + R_sqrt * randn(3,1);
        particles(3,i) = wrapToPi_local(particles(3,i));
    end

    %% Particle weighting update (robot measurement and obstacle penalty)
    for i = 1:N
        err = z - particles(:,i);
        err(3) = wrapToPi_local(err(3));

        w_meas = exp(-0.5 * err.' * invQ * err);
        w_obs  = moving_obstacle_penalty( ...
            particles(:,i), obs_meas, obs_vel_meas, obs.radii, pred_horizon);

        weights(i) = w_meas * w_obs;
    end

    weights = weights + 1e-300;
    weights = weights / sum(weights);

    % estimate robot state using particles
    x_est = particle_mean(particles, weights);

    % Effective particles
    Neff = 1 / sum(weights.^2);
    Neff_hist(end+1) = Neff;

    %% Resample
    if Neff < N/2
        idx = systematic_resample(weights);
        particles = particles(:,idx);
        weights = ones(1,N) / N;
    end

    %% store data points
    true_hist(:,end+1) = x_true;
    est_hist(:,end+1)  = x_est;
    meas_hist(:,end+1) = z;

    for j = 1:numObs
        obs_hist{j}(:,end+1) = obs.centers(:,j);
    end

    %% live plot updating
    set(hParticles, 'XData', particles(1,:), 'YData', particles(2,:));
    set(hTruePath, 'XData', true_hist(1,:), 'YData', true_hist(2,:));
    set(hEstPath,  'XData', est_hist(1,:),  'YData', est_hist(2,:));
    set(hTrueNow,  'XData', x_true(1),      'YData', x_true(2));
    set(hEstNow,   'XData', x_est(1),       'YData', x_est(2));
    set(hMeasNow,  'XData', z(1),           'YData', z(2));

    for j = 1:numObs
        [xc, yc] = circle_points(obs.centers(:,j), obs.radii(j));
        set(hObsCircle(j), 'XData', xc, 'YData', yc);
        set(hObsTrail(j),  'XData', obs_hist{j}(1,:), 'YData', obs_hist{j}(2,:));
        set(hObsMeas(j),   'XData', obs_meas(1,j), 'YData', obs_meas(2,j));
    end

    % show escape mode in title
    if escape_mode
        title('Particle Filter with Moving Obstacles - ESCAPE MODE');
    else
        title('Particle Filter with Moving Obstacles - Real-Time Trajectory');
    end

    drawnow;

    if save_video
        frame = getframe(fig1);
        writeVideo(vobj, frame);
    end

    pause(0.02);

    %% stop at/near goal
    if norm(x_true(1:2) - goal) < 1.0
        disp(['Goal reached in ', num2str(k), ' steps.']);
        disp(['Elapsed time = ', num2str(k*dt), ' seconds.']);
        break;
    end
end

%% video closing
if save_video
    close(vobj);
    disp(['Video saved as: ', video_filename]);
end

%% effective particles plot
figure('Color','w');
plot(Neff_hist, 'LineWidth', 1.8);
grid on;
xlabel('Time step');
ylabel('N_{eff}');
title('Effective Number of Particles');

%% function calls
function xnext = motion_model(x, v, gamma, L, dt)
    theta = x(3);
    xnext = [x(1) + v*cos(theta)*dt;
             x(2) + v*sin(theta)*dt;
             x(3) + (v/L)*tan(gamma)*dt];
    xnext(3) = wrapToPi_local(xnext(3));
end

function [v, gamma, min_clearance] = controller_with_moving_obstacles(x, goal, obsC, obsV, obsR, predH, escape_mode, escape_dir)

    p = x(1:2);
    theta = x(3);

    v_nom = 2.5;
    gamma_max = pi/3;
    k_gamma = 1.8;

    % Attractive goal direction
    att = goal - p;
    att = att / (norm(att) + 1e-9);

    vec = att;
    min_clearance = inf;
    nearest_idx = 1;
    nearest_clear = inf;

    influence_dist = 8.0;
    pred_influence = 10.0;
    rep_gain_cur = 2.2;
    rep_gain_pred = 1.8;

    for j = 1:length(obsR)
        c_now = obsC(:,j);
        c_pred = obsC(:,j) + obsV(:,j) * predH;

        % Current obstacle repulsion
        dvec_now = p - c_now;
        dist_now = norm(dvec_now);
        clear_now = dist_now - obsR(j);

        if clear_now < nearest_clear
            nearest_clear = clear_now;
            nearest_idx = j;
        end

        min_clearance = min(min_clearance, clear_now);

        if clear_now < influence_dist
            away_now = dvec_now / (dist_now + 1e-9);
            strength_now = rep_gain_cur * max(0, (1/max(clear_now,0.35) - 1/influence_dist));
            vec = vec + strength_now * away_now;
        end

        % Predicted obstacle repulsion
        dvec_pred = p - c_pred;
        dist_pred = norm(dvec_pred);
        clear_pred = dist_pred - obsR(j);
        min_clearance = min(min_clearance, clear_pred);

        if clear_pred < pred_influence
            away_pred = dvec_pred / (dist_pred + 1e-9);
            strength_pred = rep_gain_pred * max(0, (1/max(clear_pred,0.35) - 1/pred_influence));
            vec = vec + strength_pred * away_pred;
        end
    end

    % trapped: will turn towards one tangent direction of obstacle
    if escape_mode
        away = p - obsC(:,nearest_idx);
        if norm(away) < 1e-6
            away = [cos(theta); sin(theta)];
        end
        away_u = away / (norm(away) + 1e-9);
        tangent_u = escape_dir * [-away_u(2); away_u(1)];

        % adding goal pull and obstacle push to turn away
        vec = 1.8*tangent_u + 0.8*away_u + 0.5*att;
    end

    theta_des = atan2(vec(2), vec(1));
    e = wrapToPi_local(theta_des - theta);
    gamma = max(min(k_gamma * e, gamma_max), -gamma_max);

    % stronger repulsion in turn away
    if escape_mode
        gamma = max(min(1.15*gamma, gamma_max), -gamma_max);
        v = max(2.2, 0.9*v_nom);
    else
        %  speed 
        if min_clearance < 1.0
            v = 2;
        elseif min_clearance < 2.0
            v = 3;
        elseif min_clearance < 3.5
            v = 5;
        else
            v = v_nom;
        end
    end
end

function dir = choose_escape_direction(x, goal, obsC)
    % turn away direction based on goal proximity
    p = x(1:2);

    dists = vecnorm(obsC - p);
    [~, idx] = min(dists);

    away = p - obsC(:,idx);
    if norm(away) < 1e-6
        away = goal - p;
    end

    t_left  = [-away(2);  away(1)];
    t_right = [ away(2); -away(1)];
    g = goal - p;

    if dot(t_left, g) >= dot(t_right, g)
        dir = 1;
    else
        dir = -1;
    end
end

function w_obs = moving_obstacle_penalty(xp, obsC, obsV, obsR, predH)
    % Penalize particles that are inside or near current/predicted obstacle locations
    buffer_now = 1.8;
    buffer_pred = 2.4;
    sigma_now = 0.8;
    sigma_pred = 1.0;

    w_obs = 1.0;

    for j = 1:length(obsR)
        c_now  = obsC(:,j);
        c_pred = obsC(:,j) + obsV(:,j) * predH;

        d_now  = norm(xp(1:2) - c_now)  - obsR(j);
        d_pred = norm(xp(1:2) - c_pred) - obsR(j);

        % Hard penalty if particle is inside current or predicted obstacle region
        if d_now <= 0 || d_pred <= -0.2
            w_obs = 1e-10;
            return;
        end

        % Soft penalty near current obstacle
        if d_now < buffer_now
            w_obs = w_obs * exp(-0.5 * ((buffer_now - d_now)/sigma_now)^2);
        end

        % Soft penalty near predicted obstacle
        if d_pred < buffer_pred
            w_obs = w_obs * exp(-0.5 * ((buffer_pred - d_pred)/sigma_pred)^2);
        end
    end

    w_obs = max(w_obs, 1e-12);
end

function obs = update_obstacles(obs, dt, world)
    % Move each circular obstacle, bounce off boundaries,
    % and prevent overlap with other obstacles.

    xmin = world(1); xmax = world(2);
    ymin = world(3); ymax = world(4);

    numObs = length(obs.radii);

    % move all obstacles
    for j = 1:numObs
        obs.centers(:,j) = obs.centers(:,j) + obs.vel(:,j) * dt;
    end

    % bounce off walls
    for j = 1:numObs
        r = obs.radii(j);

        % x direction
        if obs.centers(1,j) - r < xmin
            obs.centers(1,j) = xmin + r;
            obs.vel(1,j) = -obs.vel(1,j);
        elseif obs.centers(1,j) + r > xmax
            obs.centers(1,j) = xmax - r;
            obs.vel(1,j) = -obs.vel(1,j);
        end

        % y direction
        if obs.centers(2,j) - r < ymin
            obs.centers(2,j) = ymin + r;
            obs.vel(2,j) = -obs.vel(2,j);
        elseif obs.centers(2,j) + r > ymax
            obs.centers(2,j) = ymax - r;
            obs.vel(2,j) = -obs.vel(2,j);
        end
    end

    % prevent obstacle-obstacle overlap
    buffer = 0.3;

    for i = 1:numObs-1
        for j = i+1:numObs
            c1 = obs.centers(:,i);
            c2 = obs.centers(:,j);

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
                obs.centers(:,i) = obs.centers(:,i) - 0.5 * overlap * dir;
                obs.centers(:,j) = obs.centers(:,j) + 0.5 * overlap * dir;

                % collision reaction
                obs.vel(:,i) = -obs.vel(:,i);
                obs.vel(:,j) = -obs.vel(:,j);
            end
        end
    end
end

function idx = systematic_resample(w)
    N = length(w);
    positions = ((0:N-1) + rand(1))/N;
    c = cumsum(w);
    idx = zeros(1,N);

    i = 1;
    j = 1;
    while i <= N
        if positions(i) < c(j)
            idx(i) = j;
            i = i + 1;
        else
            j = j + 1;
        end
    end
end

function a = wrapToPi_local(a)
    a = mod(a + pi, 2*pi) - pi;
end

function x_est = particle_mean(particles, weights)
    x_est = zeros(3,1);
    x_est(1) = sum(weights .* particles(1,:));
    x_est(2) = sum(weights .* particles(2,:));

    c = sum(weights .* cos(particles(3,:)));
    s = sum(weights .* sin(particles(3,:)));
    x_est(3) = atan2(s, c);
end

function [xc, yc] = circle_points(center, radius)
    ang = linspace(0, 2*pi, 200);
    xc = center(1) + radius*cos(ang);
    yc = center(2) + radius*sin(ang);
end