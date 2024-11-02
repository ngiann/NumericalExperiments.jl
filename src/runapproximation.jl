function setup_random_number_generators(seed)

    # set up random number generators

    rng = MersenneTwister(seed)

    rng2 =  MersenneTwister(seed)

    samplerng() = MersenneTwister(round(Int64, 10000 * rand(rng2)))

    return rng, samplerng

end



function run_approximation(elboapprox; iterations = 10_000, repeats = 10, seed = 1)

    
    rng, samplerng = setup_random_number_generators(seed)


    function fit_strategy(::ElboDiag)

        local p = cmaesmaximise_elbo(elboapprox, [randn(rng, 6); ones(6)], iterations = 500, rng = samplerng())

        maximise_elbo(elboapprox, getsolution(p), iterations = iterations)

    end

    function fit_strategy(::ElboMvi)

        local p = bbmaximise_elbo(elboapprox, [randn(rng, 6); ones(6)], iterations = 500, rng = samplerng())

        maximise_elbo(elboapprox,  getsolution(p), iterations = iterations)

    end

    function fit_strategy(::ElboFull)

        local p = bbmaximise_elbo(elboapprox, [randn(rng, 6); vec(1.0*Matrix(I,6,6))], iterations = 500, rng = samplerng())
        
        maximise_elbo(elboapprox, p, iterations = iterations)

    end


    function fit_strategy(::ElboMviExt)

        local p = bbmaximise_elbo(elboapprox, [randn(rng, 6); ones(6); 0], iterations = 500, rng = samplerng())

        local res = maximise_elbo(elboapprox, getsolution(p), iterations = iterations)

        for _ in 2:10

            res = maximise_elbo(elboapprox, getsolution(res), iterations = iterations)

        end

        return res

    end


    results = [fit_strategy(elboapprox) for _ in 1:repeats]

    bestindex = argmin([ELBOfyUtilities.getminimum(r) for r in results])

    best_result = results[bestindex]

    return getsolution(best_result) # return best variational parameters

end