#Pkg.activate("Project_Thesis_2324.toml")
#Pkg.instantiate()

#push!(LOAD_PATH, "./MPSKit.jl/src")

# dir = "C:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\SanderDM_Thesis_2324\\MPSKit.jl"
# include.(filter(contains(r".jl$"), readdir(dir; join=true)))

#push!(LOAD_PATH, raw"C:\Users\Sande\Documents\School\0600 - tweesis\Code\MPSKit.jl")

#include("files_from_MPSKit_Daan/multipliedoperator.jl") #src/operators
#include("files_from_MPSKit_Daan/sumofoperators.jl") #src/operators

# import MPSKit
# using MPSKit

#include("MPSKit.jl")
using MPSKit
# using MPSKit_Daan
using MPSKitModels
#include("c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\MPSKit.jl")
using TensorKit
#using MPSKitModels
using KrylovKit
using Glob
import Main
using Main

dir = "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src"

# include("includes.jl")

#include.(filter(contains(r".jl$"), readdir(dir; join=true)))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\utility"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\states"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators\\densempo"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators\\sparsempo"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\operators"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\transfermatrix"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\environments"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms\\groundstate"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms\\approximate"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms\\changebonds"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms\\excitation"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms\\propagator"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms\\statmech"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKit.jl\\src\\algorithms\\timestep"))

# dir = "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKitModels.jl\\src"
# #include.(filter(contains(r".jl$"), readdir(dir; join=true)))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKitModels.jl\\src\\lattices"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKitModels.jl\\src\\models"))
# foreach(include, glob("*.jl", "c:\\Users\\Sande\\Documents\\School\\0600 - tweesis\\Code\\MPSKitModels.jl\\src\\operators"))

# include("files_from_MPSKit_Daan/multipliedoperator.jl") #src/operators
# include("files_from_MPSKit_Daan/sumofoperators.jl") #src/operators
# include("files_from_MPSKit_Daan/extra_environments.jl") # random stuff

# include("files_from_MPSKit_Daan/defaults.jl") # random stuff

# include("files_from_MPSKit_Daan/environments/abstractinfenv.jl") # random stuff
# include("files_from_MPSKit_Daan/environments/mpohaminfenv.jl") # random stuff


# include("environments/lazylincocache.jl")


# FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^10)


function f_trivial(t)
	return 1.0
end

# some variables
N = 30
g = 0.5
f(t) = 0.1t
dt = 0.01
t_end = 0.1

gs = fill(g,N)
gs[15] = 0.7 # an impurity in the middle

# the initial state for the time_evolution
Hgs = transverse_field_ising(g=0.0);
ground = InfiniteMPS(ℂ^2,ℂ^20);

ground,envs,_ = find_groundstate(ground,Hgs,VUMPS(maxiter=200, tol_galerkin = 1e-8))

# set up window stuff
Ψ = WindowMPS(ground,N); # state is a windowMPS

X = σˣ()
xWindow = Window(X,X,X)

TmOp = TimedOperator(Hgs, f)

println(typeof(ground))
println(typeof(repeat(ground,10)))
println(typeof(Ψ))
println(typeof(Window(ground,repeat(ground,10),ground)))
println(typeof(Window(Hgs,repeat(Hgs,10),Hgs)))

windowE = environments(Ψ, Window(Hgs,Hgs,Hgs))

windowE = environments(Ψ, Window(Hgs,Hgs,TimedOperator(Hgs,f))) # werkt
windowE = environments(ground, SumOfOperators([Hgs, Hgs])) # werkt
#windowE = environments(Ψ, SumOfOperators([Window(Hgs,Hgs,Hgs), Window(Hgs,Hgs,Hgs)])) # werkt niet
windowE = environments(Ψ, Window(SumOfOperators([Hgs, Hgs]),SumOfOperators([Hgs, Hgs]),SumOfOperators([Hgs, Hgs]))) # werkt
windowE = environments(Ψ, Window(SumOfOperators([Hgs, TimedOperator(Hgs)]),SumOfOperators([Hgs, TimedOperator(Hgs)]),SumOfOperators([Hgs, TimedOperator(Hgs)]))) # werkt
# WindowE = environments(Ψ, Window(Hgs,Hgs,SumOfOperators([Hgs, TimedOperator(Hgs,f)]))); # werkt niet
# windowE = environments(Ψ, Window(Hgs,Hgs,SumOfOperators([Hgs, Hgs]))) # werkt niet

# left Hamiltonian # H(t) = -∑_{<i,j>} Z_i Z_j - f(t) ∑_{<i>} g * X_i
# H_left1 = transverse_field_ising(;g=0.0);
# H_left2 = @mpoham sum(i->-g*X{i},vertices(InfiniteChain(1)))

# # middle Hamiltonian # H(t) = -∑_{<i,j>} Z_i Z_j - f(t) ∑_{<i>} g_i * X_i
# H_mid1 = repeat(H_left1,N)
# H_mid2 = @mpoham sum(i->-gs[i]*X{i},vertices(InfiniteChain(N)));
# #Ht_mid = TimedOperator(H_mid1,f) + TimedOperator(H_mid2,g)
# Ht_mid = SumOfOperators([H_mid1, TimedOperator(H_mid2,f)])

# #right 
# H_right1 = transverse_field_ising(;g=0.);
# H_right2 = @mpoham sum(i->-0*X{i},vertices(InfiniteChain(1)))
# Ht_right = SumOfOperators([H_right1, TimedOperator(H_right2,f)])
# #Note: despite not doing time evolution with the right infinite part,
# #      for the code to work we need Ht_right to have a similar form as the left and middle H

# WindowH = Window(Ht_left,Ht_mid,Ht_right);
# test = SumOfOperators([Window(Hgs, Hgs, Hgs), Window(TimedOperator(Hgs),TimedOperator(Hgs),TimedOperator(Hgs))])
# WindowE = environments(WindowMPS(ground,repeat(ground,N),ground), test)
# WindowE = environments(ground,Hgs);


# # underlying works
Ht_left = SumOfOperators([Hgs, Hgs])
Ht_mid = SumOfOperators([Hgs, Hgs])
Ht_right = SumOfOperators([Hgs, Hgs])


# Ht_left = SumOfOperators([Hgs, 5*Hgs])
# Ht_mid = SumOfOperators([Hgs, 3*Hgs])
# Ht_right = SumOfOperators([Hgs, Hgs])

# # underlying doesn't work
# Ht_left = SumOfOperators([Hgs, TimedOperator(Hgs,f)])
# Ht_mid = SumOfOperators([Hgs, TimedOperator(Hgs,f)])
# Ht_right = SumOfOperators([Hgs, TimedOperator(Hgs,f)])


WindowH = Window(Ht_left,Ht_mid,Ht_right);
WindowE = environments(Ψ, WindowH)

# ground,envs,_ = find_groundstate(ground,Hgs,VUMPS(maxiter=200, tol_galerkin = 1e-8))
# Ψ = WindowMPS(ground,N);
# Hgs = transverse_field_ising(g=0.0);


# H_mid_v = @mpoham sum(f(i)*Interaction_v_term for i in vertices(InfiniteChain(N)))


# test_Wind = TimedOperator(Hgs,f)
# #test = SumOfOperators([Window(Hgs, Hgs, Hgs), Window(Hgs,Hgs,Hgs)])
# #test = SumOfOperators(Window(Hgs, Hgs, Hgs))
# #test = SumOfOperators([Window(Hgs, Hgs, Hgs)])
# test = SumOfOperators(Hgs) # werkt
# test = Window(Hgs, Hgs, Hgs) # werkt
# test = SumOfOperators(Window(Hgs,Hgs,Hgs)) # werkt niet
# test = Window(SumOfOperators(Hgs), SumOfOperators(Hgs), SumOfOperators(Hgs)) # werkt
# test = Window(SumOfOperators([Hgs,Hgs]), SumOfOperators([Hgs,Hgs]), SumOfOperators([Hgs,Hgs])) # werkt
# test = Window(SumOfOperators([Hgs,TimedOperator(Hgs,f)]), SumOfOperators([Hgs,Hgs]), SumOfOperators([Hgs,Hgs])) # werkt
# test = Window(SumOfOperators([Hgs,TimedOperator(Hgs,f)]), H_mid_v, SumOfOperators([Hgs,Hgs])) # werkt

# environments(Ψ,test)



# WindowH = SumOfOperators([Window(H_without_v,H_without_v,H_without_v), Window(0*Interaction_v_term, TimedOperator(Interaction_v_term_middle,f), TimedOperator(Interaction_v_term,f))]) 
# environments(Ψ,WindowH)

# WindowE = environments(Ψ,Window(Hgs,Hgs,Hgs)); # werkt
# WindowE = environments(Ψ,Window(Hgs,Hgs,TimedOperator(Hgs,f))); # werkt
# WindowE = environments(Ψ,Window(Hgs,Hgs,Hgs+Hgs)); # werkt
# WindowE = environments(Ψ, SumOfOperators([Hgs,Hgs])); # werkt
# WindowE = environments(Ψ,Window(Hgs,Hgs,SumOfOperators([Hgs,Hgs]))); # werkt niet
# WindowE = environments(Ψ,Window(Hgs,Hgs,SumOfOperators(TimedOperator(Hgs,f)))); #werkt niet
#WindowE = environments(Ψ,WindowH);

# Einit = expectation_value(Ψ,WindowH,WindowE)
# Xinit = map(i->expectation_value(window,xWindow,i),0:N+1)

tobesaved = []
t_span    = 0:dt:t_end
#alg       = TDVP(finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,tobesaved,xWindow,g,gs,0.0))
alg       = TDVP()
Ψt = copy(Ψ)

#window_dt,WindowE = time_evolve!(Ψt,WindowH,t_end,dt,alg;verbose=true,leftevolve,rightevolve=true)
for i = 1:100
    # println("Started for i = $(i)")
    # Ht_left = SumOfOperators([Hgs, f(i*dt)*Hgs])
    # Ht_mid = SumOfOperators([Hgs, f(i*dt)*Hgs])
    # Ht_right = SumOfOperators([Hgs, Hgs])
    # WindowH = Window(Ht_left,Ht_mid,Ht_right);
    # WindowE = environments(Ψ, WindowH)
    
    # does not work with rightevolve = true
    # does not work with TimedOperator()
    window_dt,WindowE = timestep!(Ψt,WindowH,t_end,dt,alg,windowE;leftevolve=true,rightevolve=false)
end

@assert false
#Hv = onsiteX();
Hramp_left = H0 + TimedOperator(hxl_final*Hv,rampfun);
#@show typeof(Hramp_left)

#window
H0mid = repeat(H0,N);
Hvmid = repeat(Hv,N);
#@show gs_final[1:5]
for (i,hxi) in enumerate(hxs_final)
    #not true in general, need H[i][1:end-1,end]
    Hvmid[i][1,end] *= hxi
end
Hramp = H0mid + TimedOperator(Hvmid,rampfun);
#@show typeof(Hramp)

#right
Hramp_right = H0 + TimedOperator(hxr_final*Hv,rampfun);

WindowH = Window(Hramp_left,Hramp,Hramp_right)
WindowE = environments(window,WindowH)
x  = collect(0:N+1)
zs = map(i->expectation_value(window,sxWindow,i),x)
hs = [hxl_final*rampfun(0.),(rampfun(0.).*hxs_final)...,hxr_final*rampfun(0.)]
tobesaved = [[0.,expectation_value(window,WindowH(0.),WindowE),zs,hs]]

t_span    = 0:dt:t_end
alg       = TDVP(integrator = integrator, finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,sxWindow,tobesaved,hxl_final,hxs_final,hxr_final,rampfun))
window_dt = copy(window)



window_dt,WindowE = time_evolve!(window_dt,WindowH,t_span,alg,WindowE;verbose=true)
