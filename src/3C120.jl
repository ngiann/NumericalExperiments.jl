function setup_3C120_joint_loglikel()

    tobs, yobs, σobs = GPCCData.readdataset(source="3C120")

    logp, pred, unpack = gpccloglikelihood(tobs, yobs, σobs, kernel = GPCC.matern32, maxdelay = 50)

end


function run_3C120_diag(; iterations = 30_000, repeats = 10, nsamples = 0, rng = MersenneTwister(1))
    
    logp, = setup_3C120_joint_loglikel()

    function fit_approximation()

        local elbodiag = elbofy_diag(logp, 6, nsamples, parallel = true)

        local p = [randn(rng, 6); 0.1*ones(6)]

        maximise_elbo(elbodiag, getsolution(p), iterations = iterations, show_trace = true)

    end

    @showprogress map( _ -> fit_approximation(), 1:repeats)

end


function run_3C120_full(; iterations = 30_000, repeats = 10, nsamples = 0, rng = MersenneTwister(1))
    
    logp, = setup_3C120_joint_loglikel()
    
    function fit_approximation()

        local elbofull = elbofy_full(logp, 6, nsamples)

        local p = [randn(rng, 6); vec(0.1*Matrix(I,6,6))]

        maximise_elbo(elbofull, getsolution(p), iterations = iterations, show_trace = true)

    end

    @showprogress map( _ -> fit_approximation(), 1:repeats)

end


function run_3C120_mvi(; iterations = 30_000, repeats = 10, nsamples = 0, rng = MersenneTwister(1))
    
    logp, = setup_3C120_joint_loglikel()

    function fit_approximation()

        local resphere = let

            local elbosphere = elbofy_sphere(logp, 6, nsamples)
  
            maximise_elbo(elbosphere, iterations = iterations, show_trace = true)

        end

        local V = geteigenvectors(logp, resphere.minimizer[1:6])

        local elbomvi = elbofy_mvi(logp, V, nsamples)

        local p = [randn(rng, 6); 0.1*ones(6)]

        maximise_elbo(elbomvi, getsolution(p), iterations = iterations, show_trace = true)

    end

    @showprogress map( _ -> fit_approximation(), 1:repeats)

end


function run_3C120_mvi_ext(; iterations = 30_000, repeats = 10, nsamples = 0, rng = MersenneTwister(1))
    
    logp, = setup_3C120_joint_loglikel()

    function fit_approximation()

        local p = [randn(rng, 6); ones(6); 0]

        local elbomviext = elbofy_mvi_ext(logp, 0.1*Matrix(I,6,6), nsamples)

        local res = maximise_elbo(elbomviext, getsolution(p), iterations = iterations, show_trace = true)

        local prvfitness = res.minimum
        
        local tol = 1e-4

        for _ in 2:10

            elbomviext, res = updatecovariance(elbomviext, res)

            res = maximise_elbo(elbomviext, getsolution(res), iterations = iterations)

            if abs(res.minimum - prvfitness)<tol
                
                break

            end

            prvfitness = res.minimum
            
        end

        res

    end

    @showprogress map( _ -> fit_approximation(), 1:repeats)
    
end


function run_3C120_skew(; iterations = 30_000, repeats = 10, nsamples = 0, rng = MersenneTwister(1))
    
    logp, = setup_3C120_joint_loglikel()

    function fit_approximation()

        local elboskew = elbofy_skewdiag(logp, 6, nsamples)

        maximise_elbo(elboskew, randn(rng, numparam(elboskew)), iterations = iterations, show_trace = true)

    end

    @showprogress map( _ -> fit_approximation(), 1:repeats)

end