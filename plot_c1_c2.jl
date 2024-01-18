using Plots
gr()

m = 0.5
v = 0.0

function _Pk_matrix(k, m, v)
    c1 = (1im/2)*(1-exp(im*k))
    c2 = (m+(v/2)*sin(k))
    norm = sqrt(abs(c1)^2+abs(c2)^2)
    return (c1/norm, c2/norm)
end

k_values = range(-pi,pi,1000)

c1_valuesr = [real(_Pk_matrix(k, m, v)[1]) for k in k_values]
c2_valuesr = [real(_Pk_matrix(k, m, v)[2]) for k in k_values]
c1_valuesi = [imag(_Pk_matrix(k, m, v)[1]) for k in k_values]
c2_valuesi = [imag(_Pk_matrix(k, m, v)[2]) for k in k_values]

p = plot(k_values, c1_valuesr, label="c1 values, real")
plot!(k_values, c2_valuesr, label="c2 values, real")
plot!(k_values, c2_valuesi, label="c2 values, imag")
plot!(k_values, c2_valuesi, label="c2 values, imag")

display(p)