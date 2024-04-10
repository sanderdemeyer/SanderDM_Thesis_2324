using KrylovKit
using TensorKit
using MPSKitModels, TensorKit, MPSKit

# Onderstaande code maakt op 4 verschillende manieren een mpoham aan, gebaseerd op een lijst dat de factoren per site opslaat.
# We zouden op het eerste zicht verwachten dat MPO0 = MPO1 = MPO2 = MPO3, maar de keys die op 0 worden gezet, verschillen.

N = 5

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
trivspace = U1Space(0 => 1)
Z = TensorMap([1.0+0.0im 0.0+0.0im;0.0+0.0im -1.0+0.0im;], trivspace ⊗ pspace, pspace ⊗ trivspace)
Z2 = TensorMap([1.0+0.0im 0.0+0.0im;0.0+0.0im -1.0+0.0im;], pspace, pspace)

my_next_nearest_neighbours(chain::InfiniteChain) = map(v -> (v, v + 1, v + 2), vertices(chain))

lijst0 = (1:N)*0.0
lijst = (1:N)

# make a mpoham with a list 'lijst', which contains only zeroes
MPO0 = @mpoham sum(lijst0[i]*Z2{i}*Z2{j}*Z2{k} for (i,j,k) in my_next_nearest_neighbours(InfiniteChain(N)));

# make a mpoham with a list 'lijst', not containing zeroes
MPO_base = @mpoham sum(lijst[i]*Z2{i}*Z2{j}*Z2{k} for (i,j,k) in my_next_nearest_neighbours(InfiniteChain(N)));

MPO1 = @mpoham (0.0) * sum(lijst[i]*Z2{i}*Z2{j}*Z2{k} for (i,j,k) in my_next_nearest_neighbours(InfiniteChain(N)));

# multiply the MPO1 with 0, we would expect that MPO0 = MPO2
MPO2 = 0*copy(MPO_base);

# Explicit multiplication with 0
MPO3 = repeat(MPOHamiltonian([Z, Z, Z]),N);
for i = 1:N
    for k = 1:MPO3.odim-1
        MPO3[i][k,MPO3.odim] *= lijst0[i];
    end
end

println(collect(keys(MPO_base[1])))
println(collect(keys(MPO0[1])))
println(collect(keys(MPO1[1])))
println(collect(keys(MPO2[1])))
println(collect(keys(MPO3[1])))

println("For MPO0 and MPO1, (1,2) is set to zero.")
println("For MPO2 and MPO3, (3,4) is set to zero.")