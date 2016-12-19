function [f,u,v,e] = IntensityMotionRecovery(x,y,pol,time,triggers)

%% initialization
%%% First, specify the 3D cube: 128 X 128 X length(time)
option.nx = 128;
option.ny = 128;
pol = double(pol);
pol(pol==0) = -1;
T = 8*10^3; % 30 ms. If we can finish computing in 30ms, we achieve real-time, in terms of 30fps
T0 = 3*10^4;
delta_t = 0.8*10^3; % temporal cell in a sliding window
time = time-time(1); % offseting the time axis
n_events = 1:length(time);
option.nt = round(T/delta_t); % grids in temporal domain

%%% Second, specify the initial value
f = zeros(option.ny,option.nx,option.nt);
u = zeros(option.ny,option.nx,option.nt);
v = zeros(option.ny,option.nx,option.nt);
e = u;
phi = u;

%%% Third, we extract the events within the spatio-temporal cube
for ii = 1:option.nt
    idx = find(n_events>=T0 + (delta_t*(ii-1)+1) & n_events<=T0 + (delta_t*ii));
    for kk = 1:length(idx)
        e(128-y(idx(kk)),128-x(idx(kk)),ii) = pol(idx(kk));
    end
end;
%%% Fourth, specify hyper-parameters: regularization weights and
%%% Charbonnier parameters. Specify numerical methods: Jacobian method
%{
option.alpha_1 = 500; % weights - brightness constraint
option.alpha_2 = 10; % weights - event term
option.alpha_3 = 0.5; % weights - flow regularization
option.alpha_4 = 5; %  weights - image temporal regularization
%}
option.alpha_1 = 10;
option.alpha_2 = 10;
option.alpha_3 = 1;
%{
option.lambda_d = 0.01; % Charbonnier - event term
option.lambda_b = 0.0001; % Charbonnier - brightness term
option.lambda_f = 0.001; % Charbonnier - intensity spatio smoothness term
option.lambda_o = 0.01; % Charbonnier - flow smoothness term
option.lambda_t = 0.001; % Charbonnier - intensity temporal smoothness term
%}
option.lambda_d = 0.1;
option.lambda_phi = 0.1;
option.lambda_t = 0.001;
option.lambda_f = 0.001;
option.solver = 'jacobian';
option.ONLY_INTENSITY_RECOVERY = true;
option.max_iter_inner = 1;
option.max_iter_outer = 3000;
option.max_iter_inner2 = 1

%% main-loop
for tt = 1:option.max_iter_outer
    fprintf('-iteration = %i    ',tt);
    % intensity recovery
    %fprintf('--recovering intensity\n');
%    f = IntensityEstimate(e,f,u,v,option);
    f = IntensityEstimateHigherOrder(e,f,phi,option);
    % motion recovery
    %fprintf('--recovering motion\n');
    %[u,v] = MotionEstimate(e,f,u,v,option);
    fprintf('min(f) = %f    max(f) = %f\r    ',min(f(:)),max(f(:)));
    %fprintf('min(u) = %f    max(u) = %f    ',min(u(:)),max(u(:)));
    %fprintf('min(v) = %f    max(v) = %f\r',min(v(:)),max(v(:)));

end
fprintf('\n');


%{

option.n_levels = 3;
%% create image pyramid for e
for ll = 1:option.n_levels
    factor = 1/2^(option.n_levels-ll);
    ep{ll} = imresize(e,factor);
    fp{ll} = zeros(size(ep{ll}));
    up{ll} = zeros(size(ep{ll}));
    vp{ll} = zeros(size(ep{ll}));
end

for ll = 1:option.n_levels
    fprintf('-level = %i\n',ll);
    for tt = 1:option.max_iter_outer
        fprintf('--iteration = %i\r',tt);
                % intensity recovery
        %fprintf('--recovering intensity\n');
        fp{ll} = IntensityEstimate(ep{ll},fp{ll},up{ll},vp{ll},option);
        % motion recovery
        %fprintf('--recovering motion\n');
        [up{ll},vp{ll}] = MotionEstimate(fp{ll},up{ll},vp{ll},option);
    end
    fprintf('\n');

    if ll<option.n_levels
        fp{ll+1} = imresize(fp{ll},2);
        up{ll+1} = imresize(up{ll},2);
        vp{ll+1} = imresize(vp{ll},2);
    else
        f = fp{ll};
        u = up{ll};
        v = vp{ll};
    end
end
%}

end



function f = IntensityEstimate(e,f,u,v,option)
nx = option.nx;
ny = option.ny;
nt = option.nt;
alpha_1 = option.alpha_1;
alpha_2 = option.alpha_2;
alpha_3 = option.alpha_3;
alpha_4 = option.alpha_4;
lambda_d = option.lambda_d;
lambda_b = option.lambda_b;
lambda_f = option.lambda_f;
lambda_t = option.lambda_t;
hs = 1;
ht = 1;
eps = 0;
% specify the impainting weights
if option.ONLY_INTENSITY_RECOVERY==true
    alpha_1 = 0;
end

switch option.solver
    case 'jacobian'
        em = MirrorImageBoundary(e);
        rho = exp(-5*em.^2);
        %rho = 0;
        for t = 1:option.max_iter_inner
            %%% mirror image boundary
            fm = MirrorImageBoundary(f);
            um = MirrorImageBoundary(u);
            vm = MirrorImageBoundary(v);

            %%% update nonlinearity
            [fx,fy,ft] = gradient(fm);
            psi_prime_d = 1./sqrt(1+(ft-em).^2 / lambda_d^2);
            psi_prime_b = 1./sqrt(1+(fx.*um+fy.*vm+ft).^2 / lambda_b^2);
            psi_prime_f = 1./sqrt(1+(fx.^2+fy.^2) / lambda_f^2);
            psi_prime_t = 1./sqrt(1+ft.^2 / lambda_t^2);
            [~,~,data_term] = gradient(alpha_2.*psi_prime_d.*em);
            %%% solving linear equation
            % First, compute used terms
            A = alpha_1.*psi_prime_b.*um.*um;
            B = alpha_1.*psi_prime_b.*vm.*vm;
            C = alpha_1.*psi_prime_b.*um.*vm;
            D = alpha_1.*psi_prime_b.*um;
            E = alpha_1.*psi_prime_b.*vm;
            F = alpha_1.*psi_prime_b;
            G = psi_prime_f;
            H = alpha_2*(1-rho).*psi_prime_d;
            I = alpha_4*rho.*psi_prime_t;


%            Wc =alpha_4+ (A+G) + 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(circshift(A,-1,2)+circshift(G,-1,2))...
%               + (B+G) + 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(circshift(B,-1,1)+circshift(G,-1,1))... 
%               + (F+H) + 0.5*(circshift(F,1,3)+circshift(H,1,3)) + 0.5*(circshift(F,-1,3)+circshift(H,-1,3));

            Wpcc = 0.5*(circshift(A,-1,2)+circshift(G,-1,2)) + 0.5*(A+G);
            Wncc = 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(A+G);
            Wcpc = 0.5*(circshift(B,-1,1)+circshift(G,-1,1)) + 0.5*(B+G);
            Wcnc = 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(B+G);
            Wccp = 0.5*(circshift(F,-1,3)+circshift(H,-1,3)+circshift(I,-1,3)) + 0.5*(F+H+I);
            Wccn = 0.5*(circshift(F,1,3)+circshift(H,1,3)+circshift(I,1,3)) + 0.5*(F+H+I);
            Wc = eps + Wpcc+Wncc+Wcpc+Wcnc+Wccp+Wccn;

            Wcpp = circshift(E,-1,1)/4 + circshift(E,-1,3)/4;
            Wcnp = -circshift(E,-1,3)/4 - circshift(E,1,1)/4;
            Wcpn = -circshift(E,1,3)/4 - circshift(E,-1,1)/4;
            Wcnn = circshift(E,1,3)/4 + circshift(E,1,1)/4;

            Wppc = circshift(C,-1,1)/4 + circshift(C,-1,2)/4;
            Wnpc = -circshift(C,-1,1)/4 - circshift(C,1,2)/4; 
            Wpnc = -circshift(C,1,1)/4 - circshift(C,-1,2)/4;
            Wnnc = circshift(C,1,1)/4 + circshift(C,1,2)/4;

            Wpcp = circshift(D,-1,2)/4 + circshift(D,-1,3)/4;
            Wpcn = -circshift(D,-1,1)/4 - circshift(D,1,3)/4;
            Wncp = -circshift(D,1,2)/4 - circshift(D,-1,3)/4;
            Wncn = circshift(D,-1,2)/4 + circshift(D,-1,3)/4;

            fm =(Wpcc.*circshift(fm,-1,2) + Wncc.*circshift(fm,1,2) + Wcpc.*circshift(fm,-1,1)...
              + Wcnc.*circshift(fm,1,1) + Wccp.*circshift(fm,-1,3) + Wccn.*circshift(fm,1,3)...
              + Wcpp.*circshift(fm,[-1,0,-1]) + Wcnp.*circshift(fm,[1,0,-1]) + Wcpn.*circshift(fm,[-1,0,1])...
              + Wcnn.*circshift(fm,[1,0,1]) + Wppc.*circshift(fm,[-1,-1,0]) + Wnpc.*circshift(fm,[-1,1,0])...
              + Wpnc.*circshift(fm,[1,-1,0]) + Wnnc.*circshift(fm,[1,1,0]) + Wpcp.*circshift(fm,[0,-1,-1])...
              + Wpcn.*circshift(fm,[0,-1,1]) + Wncp.*circshift(fm,[0,1,-1]) + Wncn.*circshift(fm,[0,1,1])...
              - (1-rho).*data_term)./Wc;
              
            f  = fm(2:end-1,2:end-1,2:end-1);

            %%%% hard constraint: f \in [0,1]
            %f(f<0)=0;f(f>1)=1;
        end
       
    otherwise
        error('it is not implemented.');
end
end



function f = IntensityEstimate2(e,f,u,v,option)
%%% this function solves the energy functional, where the ft is totoally replaced by e
nx = option.nx;
ny = option.ny;
nt = option.nt;
alpha_1 = option.alpha_1;
alpha_2 = option.alpha_2;
alpha_3 = option.alpha_3;
alpha_4 = option.alpha_4;
lambda_d = option.lambda_d;
lambda_b = option.lambda_b;
lambda_f = option.lambda_f;
hs = 1;
ht = 1;
eps = 1e-8;
% specify the impainting weights


switch option.solver
    case 'jacobian'
        em = MirrorImageBoundary(e);
        rho = exp(-4*em.^2);
        um = MirrorImageBoundary(u);
        vm = MirrorImageBoundary(v);
        rho = 0;

        %rho = 0;
        for t = 1:option.max_iter_inner
            %%% mirror image boundary
            fm = MirrorImageBoundary(f);
            %%% update nonlinearity
            [fx,fy,~] = gradient(fm);
            psi_prime_b = 1./sqrt((fx.*um+fy.*vm+em).^2 + lambda_b^2);
            psi_prime_f = 1./sqrt((fx.^2+fy.^2) + lambda_f^2);
            
            %%% solving linear equation
            % First, compute used terms
            A = (1-rho).*alpha_1.*psi_prime_b.*um.*um;
            B = (1-rho).*alpha_1.*psi_prime_b.*vm.*vm;
            C = (1-rho).*alpha_1.*psi_prime_b.*um.*vm;
            D = (1-rho).*alpha_1.*psi_prime_b.*um;
            E = (1-rho).*alpha_1.*psi_prime_b.*vm;
            F = (1-rho).*alpha_1.*psi_prime_b;
            G = psi_prime_f;
            PP = psi_prime_b.*um.*em;
            QQ = psi_prime_b.*vm.*em;
            [PPx,~] = gradient(PP);
            [~,QQy] = gradient(QQ);
            data_term = (1-rho).*alpha_1.*(PPx+QQy);


%            Wc =alpha_4+ (A+G) + 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(circshift(A,-1,2)+circshift(G,-1,2))...
%               + (B+G) + 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(circshift(B,-1,1)+circshift(G,-1,1))... 
%               + (F+H) + 0.5*(circshift(F,1,3)+circshift(H,1,3)) + 0.5*(circshift(F,-1,3)+circshift(H,-1,3));

            Wpcc = 0.5*(circshift(A,-1,2)+circshift(G,-1,2)) + 0.5*(A+G);
            Wncc = 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(A+G);
            Wcpc = 0.5*(circshift(B,-1,1)+circshift(G,-1,1)) + 0.5*(B+G);
            Wcnc = 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(B+G);
            Wc = eps + Wpcc+Wncc+Wcpc+Wcnc;

            Wppc = circshift(C,-1,1)/4 + circshift(C,-1,2)/4;
            Wnpc = -circshift(C,-1,1)/4 - circshift(C,1,2)/4; 
            Wpnc = -circshift(C,1,1)/4 - circshift(C,-1,2)/4;
            Wnnc = circshift(C,1,1)/4 + circshift(C,1,2)/4;

            fm =(eps*128+Wpcc.*circshift(fm,-1,2) + Wncc.*circshift(fm,1,2) + Wcpc.*circshift(fm,-1,1) + Wcnc.*circshift(fm,1,1)...
              + Wppc.*circshift(fm,[-1,-1,0])...
              + Wnpc.*circshift(fm,[-1,1,0])...
              + Wpnc.*circshift(fm,[1,-1,0])...
              + Wnnc.*circshift(fm,[1,1,0])...
              + data_term)./Wc;
              
            f  = fm(2:end-1,2:end-1,2:end-1);

            %%%% hard constraint: f \in [0,1]
            %f(f<0)=0;f(f>1)=1;
        end
       
    otherwise
        error('it is not implemented.');
end
end

%%% events and non-events locations are separated.
function f = IntensityEstimate3(e,f,u,v,option)
nx = option.nx;
ny = option.ny;
nt = option.nt;
alpha_1 = option.alpha_1;
alpha_2 = option.alpha_2;
alpha_3 = option.alpha_3;
alpha_4 = option.alpha_4;
lambda_d = option.lambda_d;
lambda_b = option.lambda_b;
lambda_f = option.lambda_f;
lambda_t = option.lambda_t;
hs = 1;
ht = 1;
eps = 0;
% specify the impainting weights
if option.ONLY_INTENSITY_RECOVERY==true
    alpha_1 = 0;
end

switch option.solver
    case 'jacobian'
        em = MirrorImageBoundary(e);
        rho = exp(-5*em.^2);
        %rho = 0;
        for t = 1:option.max_iter_inner
            %%% mirror image boundary
            fm = MirrorImageBoundary(f);
            um = MirrorImageBoundary(u);
            vm = MirrorImageBoundary(v);

            %%% update nonlinearity
            [fx,fy,ft] = gradient(fm);
            psi_prime_d = 1./sqrt(1+(ft-em).^2 / lambda_d^2);
            psi_prime_b = 1./sqrt(1+(fx.*um+fy.*vm+ft).^2 / lambda_b^2);
            psi_prime_f = 1./sqrt(1+(fx.^2+fy.^2) / lambda_f^2);
            psi_prime_t = 1./sqrt(1+ft.^2 / lambda_t^2);
            [~,~,data_term] = gradient(alpha_2.*psi_prime_d.*em);
            %%% solving linear equation
            % First, compute used terms
            A = alpha_1.*psi_prime_b.*um.*um;
            B = alpha_1.*psi_prime_b.*vm.*vm;
            C = alpha_1.*psi_prime_b.*um.*vm;
            D = alpha_1.*psi_prime_b.*um;
            E = alpha_1.*psi_prime_b.*vm;
            F = alpha_1.*psi_prime_b;
            G = psi_prime_f;
            H = alpha_2*(1-rho).*psi_prime_d;
            I = alpha_4*rho.*psi_prime_t;


%            Wc =alpha_4+ (A+G) + 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(circshift(A,-1,2)+circshift(G,-1,2))...
%               + (B+G) + 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(circshift(B,-1,1)+circshift(G,-1,1))... 
%               + (F+H) + 0.5*(circshift(F,1,3)+circshift(H,1,3)) + 0.5*(circshift(F,-1,3)+circshift(H,-1,3));

            Wpcc = 0.5*(circshift(A,-1,2)+circshift(G,-1,2)) + 0.5*(A+G);
            Wncc = 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(A+G);
            Wcpc = 0.5*(circshift(B,-1,1)+circshift(G,-1,1)) + 0.5*(B+G);
            Wcnc = 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(B+G);
            Wccp = 0.5*(circshift(F,-1,3)+circshift(H,-1,3)+circshift(I,-1,3)) + 0.5*(F+H+I);
            Wccn = 0.5*(circshift(F,1,3)+circshift(H,1,3)+circshift(I,1,3)) + 0.5*(F+H+I);
            Wc = eps + Wpcc+Wncc+Wcpc+Wcnc+Wccp+Wccn;

            Wcpp = circshift(E,-1,1)/4 + circshift(E,-1,3)/4;
            Wcnp = -circshift(E,-1,3)/4 - circshift(E,1,1)/4;
            Wcpn = -circshift(E,1,3)/4 - circshift(E,-1,1)/4;
            Wcnn = circshift(E,1,3)/4 + circshift(E,1,1)/4;

            Wppc = circshift(C,-1,1)/4 + circshift(C,-1,2)/4;
            Wnpc = -circshift(C,-1,1)/4 - circshift(C,1,2)/4; 
            Wpnc = -circshift(C,1,1)/4 - circshift(C,-1,2)/4;
            Wnnc = circshift(C,1,1)/4 + circshift(C,1,2)/4;

            Wpcp = circshift(D,-1,2)/4 + circshift(D,-1,3)/4;
            Wpcn = -circshift(D,-1,1)/4 - circshift(D,1,3)/4;
            Wncp = -circshift(D,1,2)/4 - circshift(D,-1,3)/4;
            Wncn = circshift(D,-1,2)/4 + circshift(D,-1,3)/4;

            fm =(Wpcc.*circshift(fm,-1,2) + Wncc.*circshift(fm,1,2) + Wcpc.*circshift(fm,-1,1)...
              + Wcnc.*circshift(fm,1,1) + Wccp.*circshift(fm,-1,3) + Wccn.*circshift(fm,1,3)...
              + Wcpp.*circshift(fm,[-1,0,-1]) + Wcnp.*circshift(fm,[1,0,-1]) + Wcpn.*circshift(fm,[-1,0,1])...
              + Wcnn.*circshift(fm,[1,0,1]) + Wppc.*circshift(fm,[-1,-1,0]) + Wnpc.*circshift(fm,[-1,1,0])...
              + Wpnc.*circshift(fm,[1,-1,0]) + Wnnc.*circshift(fm,[1,1,0]) + Wpcp.*circshift(fm,[0,-1,-1])...
              + Wpcn.*circshift(fm,[0,-1,1]) + Wncp.*circshift(fm,[0,1,-1]) + Wncn.*circshift(fm,[0,1,1])...
              - (1-rho).*data_term)./Wc;
              
            f  = fm(2:end-1,2:end-1,2:end-1);

            %%%% hard constraint: f \in [0,1]
            %f(f<0)=0;f(f>1)=1;
        end
       
    otherwise
        error('it is not implemented.');
end
end


%%% higher order regularization (second order) in temporal domain. It can be regarded as relaxation of non-quadratic spline model
function f = IntensityEstimateHigherOrder(e,f,phi,option)

alpha_1 = option.alpha_1;
alpha_2 = option.alpha_2;
alpha_3 = option.alpha_3;

lambda_d = option.lambda_d;
lambda_phi = option.lambda_phi;
lambda_t = option.lambda_t;
lambda_f = option.lambda_f;


switch option.solver
    case 'jacobian'
       em = MirrorImageBoundary(e);
       c = abs(em);

       for t = 1:option.max_iter_inner
       
       %%% mirror image boundary
       fm = MirrorImageBoundary(f);
       phim = MirrorImageBoundary(phi);

       %%% update non-linearity
       phim_t = circshift(phim,[0,0,-1])-phim; %%%% this is on the grid (i,j,k+1/2);
       psi_prime_d = 1./sqrt(1 + (phim-em).^2./lambda_d^2); %%%% on the grid (i,j,k)
       psi_prime_phi = 1./sqrt(1 + (phim_t).^2./lambda_phi^2); %%%% on the grid (i,j,k+1/2)
       [~,~,ft] = gradient(fm);
       psi_prime_t = 1./sqrt(1 + (ft-phim).^2./lambda_d^2); %%%% this is on the grid (i,j,k)
       fx = circshift(fm,[0,-1,0])-fm; %%%% on grid (i+1/2,j,k)
       fy = circshift(fm,[-1,0,0])-fm; %%%% on grid (i,j+1/2,k)
       psi_prime_f = 1./sqrt(1 + ( 0.5*fx.^2+0.5*circshift(fx,[0,1,0]).^2 + 0.5*fy.^2+0.5*circshift(fy,[1,0,0]).^2   ).^2./lambda_d^2); %%%% on grid (i,j,k)


       %%% loop inner-2
       for tt = 1:option.max_iter_inner2
           data_term_phi = alpha_2*psi_prime_t.*ft + c.*psi_prime_d.*em;
           W_phi_ccp = alpha_1*psi_prime_phi;
           W_phi_ccn = alpha_1*circshift(psi_prime_phi,[0,0,1]);
           phim = (data_term_phi + W_phi_ccp.*circshift(phim,[0,0,-1]) + W_phi_ccn.*circshift(phim,[0,0,1]) )./(W_phi_ccp+W_phi_ccn+c.*psi_prime_d+alpha_2.*psi_prime_t);
           
           W_f_pcc = alpha_3*( 0.5*circshift(psi_prime_f,[0,-1,0])+ 0.5*psi_prime_f);
           W_f_ncc = alpha_3*( 0.5*circshift(psi_prime_f,[0,1,0])+ 0.5*psi_prime_f);
           W_f_cpc = alpha_3*( 0.5*circshift(psi_prime_f,[-1,0,0])+ 0.5*psi_prime_f);
           W_f_cnc = alpha_3*( 0.5*circshift(psi_prime_f,[1,0,0])+ 0.5*psi_prime_f);
           W_f_ccp = alpha_2*( 0.5*circshift(psi_prime_t,[0,0,-1])+ 0.5*psi_prime_t);
           W_f_ccn = alpha_2*( 0.5*circshift(psi_prime_t,[0,0,1])+ 0.5*psi_prime_t);
           data_term_f = W_f_ccn.*( 0.5*phim + 0.5*circshift(phim,[0,0,1])) - W_f_ccp.*( 0.5*phim + 0.5*circshift(phim,[0,0,-1]));
           cc = W_f_pcc + W_f_ncc + W_f_cpc + W_f_cnc + W_f_ccp + W_f_ccn;
           fm = (data_term_f ...
                + W_f_pcc.*circshift(fm,[0,-1,0]) ...
                + W_f_ncc.*circshift(fm,[0,1,0]) ...
                + W_f_cpc.*circshift(fm,[-1,0,0]) ...
                + W_f_cnc.*circshift(fm,[1,0,0]) ...
                + W_f_ccp.*circshift(fm,[0,0,-1]) ...
                + W_f_ccn.*circshift(fm,[0,0,1])) ./cc;

           phi = phim(2:end-1,2:end-1,2:end-1);
           f = fm(2:end-1,2:end-1,2:end-1);
       end


       end

    otherwise
        error('no implementation.');
end


end











       

       





 


function [u,v] = MotionEstimate(e,f,u,v,option)

alpha_1 = option.alpha_1;
alpha_3 = option.alpha_3;
lambda_o = option.lambda_o;
lambda_b = option.lambda_b;
hs = 1;
ht = 1;


switch option.solver
    case 'jacobian'
       fm = MirrorImageBoundary(f);
       em = MirrorImageBoundary(e);
       rho = exp(-4*em.^2);
       [fx, fy, ~] = gradient(fm);
       ft = em;
        rho = 0; 
       for t = 1:option.max_iter_inner

            % mirror boundary condition
            um = MirrorImageBoundary(u);
            vm = MirrorImageBoundary(v);
                        % update nonlinearity
            [ux, uy, ut] = gradient(um);
            [vx, vy, vt] = gradient(vm);
            psi_prime_b = 1./sqrt((fx.*um+fy.*vm+ft).^2 + lambda_b^2);
            psi_prime_o = 1./sqrt((ux.^2+uy.^2+ut.^2+vx.^2+vy.^2+vt.^2).^2 + lambda_o^2);

            % solve linear equation
            J11 = (1-rho).*psi_prime_b.*fx.^2;
            J12 = (1-rho).*psi_prime_b.*fx.*fy;
            J13 = (1-rho).*psi_prime_b.*fx.*ft;
            J22 = (1-rho).*psi_prime_b.*fy.^2;
            J23 = (1-rho).*psi_prime_b.*fy.*ft;

            Wpcc = alpha_3/alpha_1 * 0.5*( psi_prime_o + circshift(psi_prime_o,[0,-1,0]));
            Wncc = alpha_3/alpha_1 * 0.5*( psi_prime_o + circshift(psi_prime_o,[0,1,0]));
            Wcpc = alpha_3/alpha_1 * 0.5*( psi_prime_o + circshift(psi_prime_o,[-1,0,0]));
            Wcnc = alpha_3/alpha_1 * 0.5*( psi_prime_o + circshift(psi_prime_o,[1,0,0]));
            Wccp = alpha_3/alpha_1 * 0.5*( psi_prime_o + circshift(psi_prime_o,[0,0,-1]));
            Wccn = alpha_3/alpha_1 * 0.5*( psi_prime_o + circshift(psi_prime_o,[0,0,1]));

            Wc_u = J11 + Wpcc + Wncc + Wcpc + Wcnc + Wccp + Wccn;
            Wc_v = J22 + Wpcc + Wncc + Wcpc + Wcnc + Wccp + Wccn;

            um = (-J12.*vm - J13 + Wpcc.*circshift(um,[0,-1,0]) + Wncc.*circshift(um,[0,1,0]) + Wcpc.*circshift(um,[-1,0,0])...
                                + Wcnc.*circshift(um,[1,0,0]) + Wccp.*circshift(um,[0,0,-1]) + Wccn.*circshift(um,[0,0,1]))./Wc_u;


            vm = (-J12.*um - J23 + Wpcc.*circshift(vm,[0,-1,0]) + Wncc.*circshift(vm,[0,1,0]) + Wcpc.*circshift(vm,[-1,0,0])...
                                + Wcnc.*circshift(vm,[1,0,0]) + Wccp.*circshift(vm,[0,0,-1]) + Wccn.*circshift(vm,[0,0,1]))./Wc_v;



            u = um(2:end-1,2:end-1,2:end-1);
            v = vm(2:end-1,2:end-1,2:end-1);
        end
        
    otherwise
        error('no other methods are implemented.');
end

end
                    
    
    


function um = MirrorImageBoundary(u)
[ny,nx,nt] = size(u);
um = zeros(ny+2,nx+2, nt+2);
um(2:end-1,2:end-1,2:end-1) = u;
um(:,:,1) = um(:,:,2);
um(:,:,end) = um(:,:,end-1);
um(:,1,:) = um(:,2,:);
um(:,end,:) = um(:,end-1,:);
um(1,:,:) = um(2,:,:);
um(end,:,:) = um(end-1,:,:);
end

