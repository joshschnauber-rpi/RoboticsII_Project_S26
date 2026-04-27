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
pred_obs = 1.0;    % obstacle prediction 

% Video saving stuff
save_video = true;
video_filename = 'pf_belief_based_obstacle_avoidance_2.mp4';
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
        % ensure obstacles start inside the world boundaries
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
    ang = 2*pi*rand;                  % random direction
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
escape_dir = 1;              % determine which way to turn

progress_window = 12;        % history of steps
progress_thresh = 0.35;      % goal distance check
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
title('PF Belief-Based Obstacle Avoidance - Real-Time Trajectory');

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
    hObsTrail(j)  = plot(obs_hist{j}(1,:), obs_hist{j}(2,:), 'Color', [0.2 0.6 0.2], 'LineStyle', '--', 'LineWidth', 1.5);
    hObsMeas(j)   = plot(obs_meas(1,j), obs_meas(2,j), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
end

legend({'Particles','True Path','Estimated Path','True Position','Estimated Position','Robot Measurement','Goal'}, 'Location','bestoutside');

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

    % PF-based control selection:
    % choose control using the full weighted particle cloud
    [v, gamma, min_clearance] = controller_pf_expected_cost(particles, weights, x_est, goal, obs_meas, obs_vel_meas, obs.radii, L, dt, pred_obs, escape_mode, escape_dir);

    % update escape state
    if escape_mode
        escape_timer = escape_timer - 1;

        if min_clearance > escape_clearance_thresh || escape_timer <= 0
            escape_mode = false;
            stuck_count = 0;
        end
    end

    % true robot movement
    x_true = motion_model(x_true, v, gamma, L, dt);
    x_true = x_true + R_sqrt * randn(3,1);
    x_true(3) = wrapToPi_local(x_true(3));

    % robot pose measurement
    z = x_true + Q_sqrt * randn(3,1);
    z(3) = wrapToPi_local(z(3));

    % particle propagation
    for i = 1:N
        particles(:,i) = motion_model(particles(:,i), v, gamma, L, dt);
        particles(:,i) = particles(:,i) + R_sqrt * randn(3,1);
        particles(3,i) = wrapToPi_local(particles(3,i));
    end

    %% Particle weighting update (measurement only)
    % obstacle avoidance influence is now in the controller, not the PF weights
    for i = 1:N
        err = z - particles(:,i);
        err(3) = wrapToPi_local(err(3));
        weights(i) = exp(-0.5 * err.' * invQ * err);
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

    % when escape mode is actually activated should hopefully register in
    % plot title
    if escape_mode
        title('PF Belief-Based Obstacle Avoidance - ESCAPE MODE');
    else
        title('PF Belief-Based Obstacle Avoidance - Real-Time Trajectory');
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

function [v_best, gamma_best, min_space_est] = controller_pf_expected_cost(particles, weights, x_est, goal, obsC, obsV, obsR, L, dt, predH, escape_mode, escape_dir)

    % new pf candidate options
    gamma_max = pi/3;
    gamma_candidates = [-gamma_max, -gamma_max/2, 0, gamma_max/2, gamma_max];

    if escape_mode
        v_candidates = [2.0, 3.0, 4.0];
    else
        v_candidates = [1.5, 2.5, 3.5];
    end

    % prediciction integer for expected-cost control
    lookahead_steps = 3;

    % Cost gains
    collision_cost = 1e4;
    w_goal = 0.8;
    w_clear = 10.0;
    w_heading = 0.6;
    w_control = 0.02;
    w_collision_mass = 2e3;

    if sum(weights) <= 0
        weights = ones(1, size(particles,2)) / size(particles,2);
    else
        weights = weights / sum(weights);
    end

    best_cost = inf;
    v_best = v_candidates(min(2,length(v_candidates)));
    gamma_best = 0;

    if escape_mode
        [tangent_u, away_u] = escape_vectors(x_est, obsC, escape_dir);
    else
        tangent_u = [0; 0];
        away_u = [0; 0];
    end

    for a = 1:length(v_candidates)
        for b = 1:length(gamma_candidates)

            v_test = v_candidates(a);
            gamma_test = gamma_candidates(b);

            J = 0;
            collision_mass = 0;

            for i = 1:size(particles,2)
                xp = particles(:,i);
                Ji = 0;
                collided = false;

                for h = 1:lookahead_steps
                    xp = motion_model(xp, v_test, gamma_test, L, dt);

                    % predict obstacle positions h steps into the future
                    t_future = min(h*dt, predH);
                    obs_pred = obsC + obsV * t_future;

                    minClear = inf;
                    for j = 1:length(obsR)
                        d = norm(xp(1:2) - obs_pred(:,j)) - obsR(j);
                        minClear = min(minClear, d);
                    end

                    if minClear <= 0
                        Ji = Ji + collision_cost + 500*h;
                        collision_mass = collision_mass + weights(i);
                        collided = true;
                        break;
                    else
                        goalCost = norm(xp(1:2) - goal);
                        clearanceCost = w_clear / (minClear + 0.35);
                        Ji = Ji + w_goal*goalCost + clearanceCost;
                    end
                end

                if ~collided
                    theta_goal = atan2(goal(2)-xp(2), goal(1)-xp(1));
                    headErr = abs(wrapToPi_local(theta_goal - xp(3)));
                    Ji = Ji + w_heading*headErr + w_control*abs(gamma_test);

                    if escape_mode
                        dir_pred = [cos(xp(3)); sin(xp(3))];
                        escapeBias = 1.5*(1 - dot(dir_pred, tangent_u)) + 0.4*(1 - dot(dir_pred, away_u));
                        Ji = Ji + escapeBias;
                    end
                end

                J = J + weights(i) * Ji;
            end

            % reject controls that put too much belief/estimation into collision
            J = J + w_collision_mass * collision_mass;

            if J < best_cost
                best_cost = J;
                v_best = v_test;
                gamma_best = gamma_test;
            end
        end
    end

    % space estimate for chosen control, based on estimated state
    x_pred_est = motion_model(x_est, v_best, gamma_best, L, dt);
    obs_pred = obsC + obsV * dt;

    min_space_est = inf;
    for j = 1:length(obsR)
        d = norm(x_pred_est(1:2) - obs_pred(:,j)) - obsR(j);
        min_space_est = min(min_space_est, d);
    end
end

function [tangent_u, away_u] = escape_vectors(x, obsC, escape_dir)
    p = x(1:2);
    dists = vecnorm(obsC - p);
    [~, idx] = min(dists);

    away = p - obsC(:,idx);
    if norm(away) < 1e-6
        away = [cos(x(3)); sin(x(3))];
    end

    away_u = away / (norm(away) + 1e-9);
    tangent_u = escape_dir * [-away_u(2); away_u(1)];
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