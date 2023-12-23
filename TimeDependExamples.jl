using Pkg
Pkg.activate("TimeDepend")
using MPSKit,MPSKitModels,TensorKit

function myfinalize(t,Ψ,H,env,tobesaved,O,gl,gs,gr,timefun)
	zs = map(i->expectation_value(Ψ,O,i),0:length(Ψ)+1)
	hs = [gl*timefun(t),(timefun(t).*gs)...,gr*timefun(t)]
	push!(tobesaved,[t,expectation_value(Ψ,H(t),env),zs,hs])
	return Ψ,env
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

# left Hamiltonian # H(t) = -∑_{<i,j>} Z_i Z_j - f(t) ∑_{<i>} g * X_i
H_left1 = transverse_field_ising(;g=0.0);
H_left2 = @mpoham sum(i->-g*X{i},vertices(InfiniteChain(1)))
Ht_left = H_left1 + TimedOperator(H_left2,f)

# middle Hamiltonian # H(t) = -∑_{<i,j>} Z_i Z_j - f(t) ∑_{<i>} g_i * X_i
H_mid1 = repeat(H_left1,N)
H_mid2 = @mpoham sum(i->-gs[i]*X{i},vertices(InfiniteChain(N)));
#Ht_mid = TimedOperator(H_mid1,f) + TimedOperator(H_mid2,g)
Ht_mid = H_mid1 + TimedOperator(H_mid2,f);

#right 
H_right1 = transverse_field_ising(;g=0.);
H_right2 = @mpoham sum(i->-0*X{i},vertices(InfiniteChain(1)));
H_right = H_right1 + TimedOperator(H_right2,f) ;
#Note: despite not doing time evolution with the right infinite part,
#      for the code to work we need Ht_right to have a similar form as the left and middle H

WindowH = Window(Ht_left,Ht_mid,H_right);
WindowE = environments(Ψ,WindowH);

Einit = expectation_value(Ψ,WindowH,WindowE)
Xinit = map(i->expectation_value(window,xWindow,i),0:N+1)

tobesaved = []
t_span    = 0:dt:t_end
alg       = TDVP(finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,tobesaved,xWindow,g,gs,0.0))
Ψt = copy(Ψ)

window_dt,WindowE = time_evolve!(Ψt,WindowH,t_span,alg,WindowE;verbose=true,rightevolve=false)




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
