function gpccloglikelihood(tarray, yarray, stdarray;  maxdelay = maxdelay, kernel = kernel, JITTER = 1e-8, rng = rng)

    #---------------------------------------------------------------------
    # Check dimensions
    #---------------------------------------------------------------------

    L = length(tarray) # number of lightcurves / bands / filters

    @assert(L == length(yarray) == length(tarray) == length(stdarray))


    #---------------------------------------------------------------------
    # Auxiliary matrices
    #---------------------------------------------------------------------

    Y = reduce(vcat, yarray)                   # concatenated fluxes

    Q = GPCC.Qmatrix(length.(tarray))               # matrix for replicating elements

    Sobs = Diagonal(reduce(vcat, stdarray).^2) # observed noise matrix


    # prior on delay

    logpriordelay = roundeduniform(0, maxdelay, 0.5)

    #  weakly informative prior on scalings α

    priorα = [fitinversegamma(μ = std(y), σ = 5) for y in yarray]

    #  weakly informative prior for shift vector b

    priorb = [Normal(mean(y), 5) for y in yarray]

    # weakly informative prior on lengthscale

    priorρ =  let 
        
        local ρ₀ =  GPCC.infercommonlengthscale(tarray, yarray, stdarray, kernel = kernel, iterations = 10000, numberofrestarts=20, initialrandom=10, ρmin = 0.1, ρmax = 200.0, verbose = false, rng = rng)

        fitinversegamma(μ = ρ₀, σ = 5)

    end

    @show priorρ
    

    #---------------------------------------------------------------------
    # Define objective as marginal log-likelihood and auxiliaries
    #---------------------------------------------------------------------

    sigmoid(x) = 1.0 / (1.0 + exp(-x))

    function unpack(params)

        @assert(3L == length(params))

        local MARK = 0

        local α = exp.(params[MARK+1:MARK+L]); MARK += L

        local b = params[MARK+1:MARK+L]; MARK += L

        local τ = [0; params[MARK+1:MARK+L-1]]; MARK += L-1

        local ρ = exp(params[MARK+1]) + 1e-2; MARK += 1

        @assert(MARK == length(params))

        return α, b, τ, ρ

    end

   
    # function objective(params) # same as one below, keep for numerical verification

    #     α, b, τ, ρ = unpack(params)
        
    #     delayedx = reduce(vcat, [x.-d for (x, d) in zip(tarray, τ)])

    #     K₁ = GPCC.covariance_unit_amplitude(kernel, ρ, delayedx) + JITTER*I

    #     A = Diagonal(Q*α)

    #     return logpdf(MvNormal(Q*b, Symmetric(A*K₁*A + Sobs)), Y)

    # end

    
    function fasterobjective(params) # same as one above, but slightly faster
        
        α, b, τ, ρ = unpack(params)

        delayedx = reduce(vcat, [x.-d for (x, d) in zip(tarray, τ)])

        K₁ = GPCC.covariance_unit_amplitude(kernel, ρ, delayedx) + JITTER*I

        A = Diagonal(Q*α)

        C = cholesky(Symmetric(A*K₁*A + Sobs)).L

        logl = -0.5*sum(abs2.(C\(Y-Q*b))) - 0.5*2*sum(log.(diag(C))) - 0.5*log(2π)*size(C,1)

        logprior =  logpriordelay(τ[2]) + logpdf(priorρ, ρ)
        
        for l in 1:L

            logprior += logpdf(priorα[l], α[l]) + logpdf(priorb[l], b[l])

        end
        
        logl + logprior

    end
      

    #---------------------------------------------------------------------
    # Functions for predicting on test data
    #---------------------------------------------------------------------

    function pred(ttest0, params)

        Ntest = length(ttest0)

        ttest = [ttest0 for _ in 1:L]

        α, b, τ, ρ = unpack(params)
        
        delayedx = reduce(vcat, [x.-d for (x, d) in zip(tarray, τ)])

        K₁ = GPCC.covariance_unit_amplitude(kernel, ρ, delayedx) + JITTER*I

        A = Diagonal(Q*α)

        KSobs = Symmetric(A*K₁*A + Sobs)
  
        Q✴  = GPCC.Qmatrix(length.(ttest))

        A✴ = Diagonal(Q✴*α)

        delayedx_test = reduce(vcat, ttest)
        
        # dimensions: N × Ntest
        k✴ = A*GPCC.covariance_unit_amplitude(kernel, ρ, delayedx, delayedx_test)*A✴#delayedCovariance(kernel, α, τ, ρ, tarray, ttest)

        # Ntest × Ntest
        c = A✴*GPCC.covariance_unit_amplitude(kernel, ρ, delayedx_test)*A✴

        # full predictive covariance
        Σpred = Symmetric(c - k✴' * (KSobs \ k✴)) #+ JITTER*I

        # predictive mean

        μpred = k✴' * (KSobs \ (Y - Q*b)) + (Q✴ * b)


        # return predictions per "band" 
        
        μ_per_band = [μpred[idx] for idx in Iterators.partition(1:L*Ntest, Ntest)]

        Σpred_per_band = [Σpred[idx,idx] for idx in Iterators.partition(1:L*Ntest, Ntest)]
      
        return μ_per_band, Σpred_per_band

    end


    fasterobjective, pred, unpack
    
    
end