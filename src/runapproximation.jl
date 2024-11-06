function setup_random_number_generators(seed)

    # set up random number generators

    rng = MersenneTwister(seed)

    rng2 =  MersenneTwister(seed)

    samplerng() = MersenneTwister(round(Int64, 10000 * rand(rng2)))

    return rng, samplerng

end



function run_approximation(elboapprox; iterations = 10_000, cmaesiterations = 500, repeats = 10, seed = 1)

    
    rng, samplerng = setup_random_number_generators(seed)


    function fit_strategy(::ELBOfy.ElboDiag)

        #local p = cmaesmaximise_elbo(elboapprox, [randn(rng, 6); ones(6)], iterations = cmaesiterations, rng = samplerng())

        local p = [randn(rng, 6); ones(6)]

        maximise_elbo(elboapprox, getsolution(p), iterations = iterations)

    end

    function fit_strategy(::ELBOfy.ElboMVI)

        #local p = cmaesmaximise_elbo(elboapprox, [randn(rng, 6); ones(6)], iterations = cmaesiterations, rng = samplerng())
        
        local p = [randn(rng, 6); ones(6)]

        maximise_elbo(elboapprox, getsolution(p), iterations = iterations)

    end

    function fit_strategy(::ELBOfy.ElboFull)

        #local p = cmaesmaximise_elbo(elboapprox, [randn(rng, 6); vec(1.0*Matrix(I,6,6))], iterations = cmaesiterations, rng = samplerng())
        
        local p = [randn(rng, 6); vec(1.0*Matrix(I,6,6))]

        maximise_elbo(elboapprox, getsolution(p), iterations = iterations)

    end


    function fit_strategy(::ELBOfy.ElboMVIExt)

        # local p = cmaesmaximise_elbo(elboapprox, [randn(rng, 6); ones(6); 0], iterations = cmaesiterations, rng = samplerng())

        local p = [randn(rng, 6); ones(6); 0]
        
        local res = maximise_elbo(elboapprox, getsolution(p), iterations = iterations)

        prvfitness = res.minimum
        tol = 1e-4

        for _ in 2:10

            elboapprox, res = updatecovariance(elboapprox, res)

            res = maximise_elbo(elboapprox, getsolution(res), iterations = iterations)

            if abs(res.minimum - prvfitness)<tol
                break
            end

            prvfitness = res.minimum
            
        end

        return res

    end


    results = [fit_strategy(elboapprox) for _ in 1:repeats]

    bestindex = argmin([ELBOfyUtilities.getminimum(r) for r in results])

    best_result = results[bestindex]

    return getsolution(best_result) # return best variational parameters

end