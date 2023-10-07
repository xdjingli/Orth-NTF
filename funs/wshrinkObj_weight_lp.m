function [x,objV] = wshrinkObj_weight_lp(x, rho, sX, isWeight, mode, p)

if isWeight == 1
    mode = 1;
end

if ~exist('mode','var')                     
    C = sqrt(sX(3)*sX(2));
end

X=reshape(x,sX);



if mode == 1
    Y=X2Yi(X,3);
elseif mode == 3
    Y=shiftdim(X, 1);     
else
    Y = X;
end

Y_hat = fft(Y,[],3);
objV = 0;


if mode == 1
    n3 = sX(2);
elseif mode == 3
    n3 = sX(1);
else
    n3 = sX(3);
end

if isinteger(n3/2)
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [u_hat,s_hat,v_hat] = svd(full(Y_hat(:,:,i)),'econ');
        
        if isWeight
            weight = C./(diag(s_hat) + eps);
            tau = rho*weight;
            s_hat = soft(s_hat,diag(tau));
        else
            tau = rho;
            s_hat=diag(s_hat);
            s_hat = solve_Lp_w(s_hat, tau, p);
            s_hat=diag(s_hat);
        end
        
        objV = objV + sum(s_hat(:));
        Y_hat(:,:,i) = u_hat*s_hat*v_hat';
        
        if i > 1
            Y_hat(:,:,n3-i+2) = conj(u_hat)*s_hat*conj(v_hat)';
            objV = objV + sum(s_hat(:));
        end
    end
    [u_hat,s_hat,v_hat] = svd(full(Y_hat(:,:,endValue+1)),'econ');
    
    if isWeight
        weight = C./(diag(s_hat) + eps);
        tau = rho*weight;
        s_hat = soft(s_hat,diag(tau));
    else
        tau = rho;
        s_hat=diag(s_hat);
        s_hat = solve_Lp_w(s_hat, tau, p);
        s_hat=diag(s_hat);
    end
    
    objV = objV + sum(s_hat(:));
    Y_hat(:,:,endValue+1) = u_hat*s_hat*v_hat';
else
    endValue = int16(n3/2+1);
    for i = 1:endValue
        [u_hat,s_hat,v_hat] = svd(full(Y_hat(:,:,i)),'econ');
        if isWeight
            weight = C./(diag(s_hat) + eps);
            tau = rho*weight;
            s_hat = soft(s_hat,diag(tau));
        else
            tau=rho;
            s_hat=diag(s_hat);
            s_hat = solve_Lp_w(s_hat, tau, p);
            s_hat=diag(s_hat);
        end
        objV = objV + sum(s_hat(:));
        Y_hat(:,:,i) = u_hat*s_hat*v_hat';
        if i > 1
            Y_hat(:,:,n3-i+2) = conj(u_hat)*s_hat*conj(v_hat)';
            objV = objV + sum(s_hat(:));
        end
    end
end

Y = ifft(Y_hat,[],3);

if mode == 1
    X = Yi2X(Y,3);
elseif mode == 3
    X = shiftdim(Y, 2);
else
    X = Y;
end

x = X(:);

end
