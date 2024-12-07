function read_concrete(;target_index=1)

    lcpath = dirname(pathof(NumericalExperiments))

    file = lcpath * "/GP/concrete.csv"

    data = readdlm(file,',',skipstart=1)


    X = data[:, 2:8]

    y = data[:, target_index+8]

    y = (y.-mean(y)) / std(y)

    X₀ = Matrix(X')

    X₁ = (X₀ .- mean(X₀,dims=1)) / std(X₀)

    return X₁, y

end