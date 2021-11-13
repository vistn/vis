function samples = individual_MCMC( n_samples )

% data
data_behavior = [27; 29; 53; 52];
% fixed_random
fixed_rand = rand( 1000000, 4 );

thinning = 60;
prop_update = 200;
factor_update = 200;
lambda1 = 1; % 0 to 1
lambda2 = 3; % 0 >
prop_factor = 1e-7;

% data_behavior = [ subjNum, rnd, trialIndex, value, took_it ];
curr_sample = [ 0; 0; 0 ];
var_temp = [0.1, 0.1, 0.1];
prop_cov = diag( var_temp );

%
log_prior_curr = cal_log_prior(curr_sample);
log_like_curr = cal_log_like(curr_sample, data_behavior, fixed_rand);
log_post_curr = log_like_curr + log_prior_curr;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
samples = zeros(length(curr_sample), n_samples);

update_times = 1;
update_accept = 0;
sample_number = 1;
accept = 0;
reject = 0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a = GetSecs;

for i = 1:n_samples*thinning
    
    %a = GetSecs;curr, prop_factor, prop_cov)
    proposal_sample = sampling_proposal(curr_sample, prop_factor*prop_cov);
    
    log_prior_prop = cal_log_prior(proposal_sample);
    log_like_prop = cal_log_like(proposal_sample, data_behavior, fixed_rand);
    log_post_prop = log_prior_prop + log_like_prop;
    
    acc = log_post_prop - log_post_curr;
    if isnan(log_post_prop)
        acc = -inf;
    end
    
    if log(rand) <= acc
        curr_sample = proposal_sample;
        log_post_curr = log_post_prop;
        %log_like_curr = log_like_prop;
        accept = accept + 1;
        update_accept = update_accept + 1;
    else
        %samples(i+1) = samples(i);
        reject = reject + 1;
    end
    
    if ~rem(i,thinning)
        samples(:,sample_number) = curr_sample;
        sample_number = sample_number + 1;
        disp(round([sample_number-1,100*accept/i]))
%         disp(log_post_curr);
%         disp(log_like_curr);
%         disp(curr_sample);
    end
        
    gamma1 = (1/(update_times)^(lambda1));
    gamma2 = lambda2*gamma1;
        
    if ~rem(i, thinning*prop_update)
        start_number = max(sample_number - 800, 1);
        vector_update = samples(:, start_number:(sample_number-1));
        prop_cov = cov(vector_update');
    end
    
    if ~rem(i,factor_update)
        log_prop_factor = log(prop_factor) + gamma2 * ((update_accept / factor_update) - 0.234);
        prop_factor = exp(log_prop_factor);
%         disp(update_accept / factor_update);
        update_accept = 0;
    end
    
    %b = GetSecs;
    %disp([i, accept])
    %duration = b-a
    
end
disp(round([sample_number,100*accept/i]))
% b = GetSecs;
%disp([i, accept, b-a])
% duration = b-a

% matlabpool close

function log_prior = cal_log_prior(sample)

if sum( sample < - 10 ) > 0 ||  sum( sample > 10 ) > 0
    prior_sigma = 0;
else
    prior_sigma = 1;
end

log_prior = 1 + log( prior_sigma );


function log_likeli = cal_log_like( params, data, fixed_rand )

mu = [0, params(1), params(2), params(3)];
count = nan(4,1);
length_rand = size( fixed_rand, 1 );

randome_picks = fixed_rand + repmat( mu, length_rand, 1 );

[~, max_i] = max( randome_picks, [], 2 );

for i = 1:4
    count(i) = sum( max_i == i );
end

pi = count/length_rand;

log_likeli = log( mnpdf( data, pi ) );


function sample_all = sampling_proposal(curr_vector, prop_cov)

sample = mvnrnd( curr_vector', prop_cov );
sample_all = sample';


