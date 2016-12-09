function [f,u,v,e] = IntensityMotionRecovery(x,y,pol,time,triggers)

%% initialization
%%% First, specify the 3D cube: 128 X 128 X length(time)
option.nx = 128;
option.ny = 128;
pol = double(pol);
pol(pol==0) = -1;
T = 5*10^3; % 30 ms. If we can finish computing in 30ms, we achieve real-time, in terms of 30fps
T0 = 3*10^4;
delta_t = 1*10^3; % temporal cell in a sliding window
time = time-time(1); % offseting the time axis
n_events = 1:length(time);
option.nt = round(T/delta_t); % grids in temporal domain

%%% Second, specify the initial value
f = zeros(option.ny,option.nx,option.nt);
u = zeros(option.ny,option.nx,option.nt);
v = zeros(option.ny,option.nx,option.nt);
e = f;


%%% Third, we extract the events within the spatio-temporal cube
for ii = 1:option.nt
    idx = find(n_events>=T0 + (delta_t*(ii-1)+1) & n_events<=T0 + (delta_t*ii));
    for kk = 1:length(idx)
        e(128-y(idx(kk)),128-x(idx(kk)),ii) = pol(idx(kk));
    end
end;

%%% Fourth, specify hyper-parameters: regularization weights and
%%% Charbonnier parameters. Specify numerical methods: Jacobian method
option.alpha_1 = 1; % weights - brightness constraint
option.alpha_2 = 10; % weights - intensity regularization
option.alpha_3 = 1; % weights - flow regularization
option.lambda_d = 0.5; % Charbonnier - data term
option.lambda_b = 2.5; % Charbonnier - brightness term
option.lambda_f = 0.005; % Charbonnier - intensity smoothness term
option.lambda_o = 0.005; % Charbonnier - flow smoothness term
option.solver = 'jacobian';

option.max_iter_inner = 1;
option.max_iter_outer = 300;


%% main-loop
for tt = 1:option.max_iter_outer
    fprintf('-iteration = %i\n',tt);
    % intensity recovery
    fprintf('--recovering intensity\n');
    f = IntensityEstimate(e,f,u,v,option);
    % motion recovery
    fprintf('--recovering motion\n');
    [u,v] = MotionEstimate(f,u,v,option);
end


end



function f = IntensityEstimate(e,f,u,v,option)
nx = option.nx;
ny = option.ny;
nt = option.nt;
alpha_1 = option.alpha_1;
alpha_2 = option.alpha_2;
alpha_3 = option.alpha_3;
lambda_d = option.lambda_d;
lambda_b = option.lambda_b;
lambda_f = option.lambda_f;
hs = 1;
ht = 1;

% specify the impainting weights
rho = MirrorImageBoundary(exp(-100*e.^2));



switch option.solver
    case 'jacobian'
        em = MirrorImageBoundary(e);

        for t = 1:option.max_iter_inner
            %%% mirror image boundary
            fm = MirrorImageBoundary(f);
            um = MirrorImageBoundary(u);
            vm = MirrorImageBoundary(v);

            %%% update nonlinearity
            [fx,fy,ft] = gradient(fm);
            psi_prime_d = 1./sqrt(1+(ft-em).^2 ./ lambda_d^2);
            psi_prime_b = 1./sqrt(1+(fx.*um+fy.*vm+ft).^2 ./ lambda_b^2);
            psi_prime_f = 1./sqrt(1+(fx.^2+fy.^2+ft.^2) ./ lambda_f^2);

            %%% solving linear equation
            % First, compute used terms
            A = alpha_1.*psi_prime_b.*um.*um;
            B = alpha_1.*psi_prime_b.*vm.*vm;
            C = alpha_1.*psi_prime_b.*um.*vm;
            D = alpha_1.*psi_prime_b.*um;
            E = alpha_1.*psi_prime_b.*vm;
            F = alpha_1.*psi_prime_b;
            G = alpha_2.*psi_prime_f;

            Wc = (A+G) + 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(circshift(A,-1,2)+circshift(G,-1,2))...
               + (B+G) + 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(circshift(B,-1,1)+circshift(G,-1,1))... 
               + (F+G) + 0.5*(circshift(F,1,3)+circshift(G,1,3)) + 0.5*(circshift(F,-1,3)+circshift(G,-1,3));

            Wpcc = 0.5*(circshift(A,-1,2)+circshift(G,-1,2)) + 0.5*(A+G);
            Wncc = 0.5*(circshift(A,1,2)+circshift(G,1,2)) + 0.5*(A+G);
            Wcpc = 0.5*(circshift(B,-1,1)+circshift(G,-1,1)) + 0.5*(B+G);
            Wcnc = 0.5*(circshift(B,1,1)+circshift(G,1,1)) + 0.5*(B+G);
            Wccp = 0.5*(circshift(F,-1,3)+circshift(G,-1,3)) + 0.5*(F+G) + (1-rho).*psi_prime_d./2;
            Wccn = 0.5*(circshift(F,1,3)+circshift(G,1,3)) + 0.5*(F+G) - (1-rho).*psi_prime_d./2;

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

            fm =( Wpcc.*circshift(fm,-1,2) + Wncc.*circshift(fm,1,2) + Wcpc.*circshift(fm,-1,1)...
              + Wcnc.*circshift(fm,1,1) + Wccp.*circshift(fm,-1,3) + Wccn.*circshift(fm,1,3)...
              + Wcpp.*circshift(fm,[-1,0,-1]) + Wcnp.*circshift(fm,[1,0,-1]) + Wcpn.*circshift(fm,[-1,0,1])...
              + Wcnn.*circshift(fm,[1,0,1]) + Wppc.*circshift(fm,[-1,-1,0]) + Wnpc.*circshift(fm,[-1,1,0])...
              + Wpnc.*circshift(fm,[1,-1,0]) + Wnnc.*circshift(fm,[1,1,0]) + Wpcp.*circshift(fm,[0,-1,-1])...
              + Wpcn.*circshift(fm,[0,-1,1]) + Wncp.*circshift(fm,[0,1,-1]) + Wncn.*circshift(fm,[0,1,1])...
              - (1-rho).*psi_prime_d.*em)./Wc;
              
            f  = fm(2:end-1,2:end-1,2:end-1);

            %%%% hard constraint: f \in [0,1]
            f = (f-min(f(:)))./(max(f(:))-min(f(:)));
        end
       
    otherwise
        error('it is not implemented.');
end
end




function [u,v] = MotionEstimate(f,u,v,option)

alpha_1 = option.alpha_1;
alpha_3 = option.alpha_3;
lambda_o = option.lambda_o;
lambda_b = option.lambda_b;
hs = 1;
ht = 1;


switch option.solver
    case 'jacobian'
        fm = MirrorImageBoundary(f);
        for t = 1:option.max_iter_inner

            % mirror boundary condition
            um = MirrorImageBoundary(u);
            vm = MirrorImageBoundary(v);

            % update nonlinearity
            [fx, fy, ft] = gradient(fm);
            [ux, uy, ut] = gradient(um);
            [vx, vy, vt] = gradient(vm);
            psi_prime_b = 1./sqrt(1+(fx.*um+fy.*vm+ft).^2 ./ lambda_b^2);
            psi_prime_o = 1./sqrt(1+(ux.^2+uy.^2+ut.^2+vx.^2+vy.^2+vt.^2).^2 ./ lambda_o^2);

            % solve linear equation
            J11 = psi_prime_b.*fx.^2;
            J12 = psi_prime_b.*fx.*fy;
            J13 = psi_prime_b.*fx.*ft;
            J22 = psi_prime_b.*fy.^2;
            J23 = psi_prime_b.*fy.*ft;

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

