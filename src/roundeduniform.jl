function roundeduniform(μ₁, μ₂, σ)

    @assert(μ₁ < μ₂)

    p₁ = Normal(μ₁, σ)

    p₂ = Normal(μ₂, σ)

    U = logpdf(p₁, μ₁)

    @assert U ≈ logpdf(p₁, μ₁)

    
    function f(x)
        
        if x <= μ₁
        
            return logpdf(p₁, x)
        
        elseif x >= μ₂

            return logpdf(p₂, x)
        end

        return U

    end

end