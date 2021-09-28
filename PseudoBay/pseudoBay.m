function [Mask_out,tf] = pseudoBay(tfr,Ncomp,M,L,div,beta,alpha,ds,Pnei,ifplot)
%
% Main algorithm: estimate the ridge position and the variance of the
% posterior distribution to propagate information to the next time sequence
%
% INPUT:
% tfr           : Time-frequency represerntation of the MCS
% Ncomp         : Number of component
% M             : Number of frequential bin
% L             : analysis window size (in bin)
% div           : Entropy choice (1 : KL | 2 : beta | 3 : Renyi)
% beta          : beta divergence hyperparameter
% alpha         : Renyi divergence hyperparameter
% ds            : variance of the random walk in the temporal model
% ifplot        : Boolean for debugging
% Pnei          : number of neighbors considered in the mask
%
%
% OUTPUT:
% Mask_out      : Mask of five bins per ridge. Modify code below to change
%                 the neighborhood


if ~exist('div', 'var')
 div=1;% default KL
end
if (~exist('beta', 'var') && (div==1))
 beta=1;% beta hyperparameter for beta divergence
end
if (~exist('alpha', 'var') && (div==2))
 alpha=0.5;%  alpha hyperparameter for Renyi divergence
end
if ~exist('ds', 'var')
 ds=3;% variance of the random walk in the temporal model
end
if ~exist('Pnei', 'var')
 Pnei=4;% default KL
end


data = transpose(abs(tfr(1:round(M/2),:))).^2; % Absolute value of the top half TFR
[Niter,N] = size(data); % Extract dimenssions

[F_mat, Fbeta_mat, IntFbeta, Falpha_mat, varF]=compFbeta_STFT(beta,alpha,M,L,N);
LFM = log(F_mat+eps);


%% Initialization
tf=zeros(Niter,Ncomp); %Array to store the means of the depth
stf=zeros(Niter,Ncomp); %Array to store the variances of the depth 

for Nc = 1:Ncomp
    M2=floor(N/2); % mean of the depth prior when initizing (this can be changed)
    S2=N^2/12; % variance of the depth prior when initizing (this can be changed) % m and s2 close to the uniform distribution
 
    %% Forward estimation
    for t=1:Niter
        Y=data(t,:); % load current frame
        % Main algorithm
        [M2,S2]=online_2D(Y+eps,LFM,Fbeta_mat,Falpha_mat,ds,M2,S2,IntFbeta,beta,alpha,div);
        % Store values
        tf(t,Nc)=round(M2);
        stf(t,Nc)=S2;
    end
    
    %% Backward estimation
    for t=Niter:-1:1
        Y=data(t,:); % load current frame
        % Main algorithm
        [M2,S2]=online_2D(Y+eps,LFM,Fbeta_mat,Falpha_mat,ds,M2,S2,IntFbeta,beta,alpha,div);
        % Store values
        tf(t,Nc)=round(M2);
        stf(t,Nc)=S2;

    end

    % remove ridge using the three sigma rule of thumb
    % Computation of the ridge to remove
    tempdata = zeros(Niter,N);
    for p = 1:Niter
        % A slighly broader window is used in practice to avoid overlapping problem
        % as well as the presence of remaining energy after ridge removal.
        % The three sigma rule of thumb can also be used to define a neighboring removal mask
        tempdata(p,:) = normpdf(1:N,tf(p,Nc),1.5*varF); % compute ridge vicinity
        tempdata(p,:) = data(p,tf(p,Nc)).*(tempdata(p,:)./max(tempdata(p,:))); % normalize
    end

    if ifplot
        figure(2)
        subplot(Ncomp,2,(Nc-1)*2+1)     
        imagesc(transpose(data))
        yticklabels({'200','100','0'})
        yticks([50,150,250])
        title('Current data')
        subplot(Ncomp,2,(Nc-1)*2+2)
        imagesc(transpose(tempdata));
        title(strcat([num2str(Nc),'th estimated ridge']))
        pause
    end
    
    
    % Update the data without the just estimated rifge
    data = max(data - tempdata, 0);
end


% Computation of the mask
Mask_out=zeros(N*Niter,Ncomp);
veccol = transpose((1:Niter)-1).*N;
for Nc = 1:Ncomp
    Mask_out(max(tf(:,Nc)+veccol-1,ones(Niter,1)),Nc)=1;
    for pn = 1:Pnei
        Mask_out(max(tf(:,Nc)+veccol-1-pn,ones(Niter,1)),Nc)=1;
        Mask_out(min(tf(:,Nc)+veccol-1+pn,(N*Niter)*ones(Niter,1)),Nc)=1; 
    end
end

Mask_out = reshape(Mask_out,[N,Niter,Ncomp]);
if Ncomp < 2
   Mask_out = sum(Mask_out,3);
   Mask_out = [Mask_out;Mask_out(end:-1:1,:)];
else
   mask2 = zeros(2*N,Niter,Ncomp);
   for i = 1:Ncomp
     mask2(:,:,i) = [Mask_out(:,:,i);Mask_out(end:-1:1,:,i)];
   end 
   Mask_out = mask2;
end





