function setup_mah_gp_log(;JITTER = 1e-8)

    X, y = read_concrete()

    D, N = size(X); @assert(length(y) == N)
    

    function logmarginal(Q, α, β)
        
        local K = Symmetric(α * calculate_covariance(X, Q) + (1/β)*I + JITTER*I)

        local C = cholesky(K).L

        local logdetK = 2*sum(log.(diag(C)))

        local diff = C\y

        -0.5*sum(abs2.(diff)) - 0.5*logdetK -0.5*size(K,1)*log(2π) - 0.5*sum(abs2.(Q))
        
    end


    logp(p) = logmarginal(unpack(p, D)...)
    
    return logp, numgparam(D), X, y

end

#--------------------------------------------------------------

numgparam(D) = D*D + 2

#--------------------------------------------------------------

function gpfit(;iterations = 1, JITTER = 1e-8)


    logp, np, X, y = setup_mah_gp_log()

    D, N = size(X); @assert(length(y) == N)

    helper(p) = -logp(p)

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1)

    result = optimize(helper, 3*randn(np), ConjugateGradient(), opt)

    Q, α, β = unpack(result.minimizer, D)

    K = Symmetric(α * calculate_covariance(X, Q) + (1/β)*I + JITTER*I)


    function predict(Xtest)

        @assert(size(Xtest, 1) == D)

        # dimensions: N × Ntest
        kB✴ = α * calculate_covariance(X, Xtest, Q)

        # Ntest × Ntest
        cB = α * calculate_covariance(Xtest, Q)

        # full predictive covariance
        Σpred = Symmetric(cB - kB✴' * (K \ kB✴) + JITTER*I)

        # predictive mean

        μpred = kB✴' * (K \ (y))

        return μpred, Σpred

    end


    result.minimum, predict, Q, α, β

end
    
#--------------------------------------------------------------

rbf(D²) = exp.(-0.5*D²)


calculate_distance(X, Q) = calculate_distance(X, X, Q)

calculate_distance(X₁, X₂, Q) = pairwise(SqMahalanobis(Q), X₁, X₂) # 


calculate_covariance(X₁, X₂, Q) = rbf(calculate_distance(X₁, X₂, Q))

calculate_covariance(X, Q) = rbf(calculate_distance(X, Q))


#--------------------------------------------------------------

function unpack(p, D)

    @assert(length(p) == numgparam(D))

    local Qroot = reshape(p[1:D*D], D, D)

    local α = softplus(p[D*D + 1])

    local β = softplus(p[D*D + 2])

    return (Qroot*Qroot') + 1e-6*I, α, β

end

#--------------------------------------------------------------

function vec2lr(v)
   
    # Calculate the size of the matrix

    n = Int((sqrt(1 + 8 * length(v)) - 1) / 2)

    @assert n * (n + 1) / 2 == length(v) "Vector length must match the number of lower triangular elements"

    # Fill the matrix using a comprehension
    idx = 0  # Index in the vector

    L = [j <= i ? (idx += 1; j == i ? softplus(v[idx]) : v[idx]) : 0.0 for i in 1:n, j in 1:n]

    return L

end

#--------------------------------------------------------------

sq_mahalanobis_distances(X, Q) = sq_mahalanobis_distances(X, X, Q)

function sq_mahalanobis_distances(X1, X2, Q)
    # Ensure inputs are consistent
    D  = size(X1, 1)
    D2 = size(X2, 1)
    @assert D == D2 "Data matrices must have the same dimensionality"
    @assert size(Q) == (D, D) "Metric Q must be a square matrix of size D x D"

    # Compute Cholesky decomposition of Q for efficiency
    # L = cholesky(Q).L

    # Transform data using the decomposition
    X1_transformed = Q *  X1  # (L' \ X1) is equivalent to (L^(-T) * X1)
    X2_transformed = Q *  X2

    # Compute pairwise squared distances
    squared_norms_X1 = sum(X1_transformed .^ 2, dims=1)  # 1xN1 row vector
    squared_norms_X2 = sum(X2_transformed .^ 2, dims=1)  # 1xN2 row vector
    return  squared_norms_X1' .+ squared_norms_X2 .- 2 .* (X1_transformed' * X2_transformed)

    # # Handle precision issues
    # dist2 .= max.(dist2, 0.0)

    # # Compute distances
    # dist_matrix = sqrt.(dist2)
    # return dist_matrix
end
