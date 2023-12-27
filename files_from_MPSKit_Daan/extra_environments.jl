environments(below,opp::TimedOperator,above,leftstart,rightstart) = environments(below,opp.op,above,leftstart,rightstart)

function environments(Ψ::WindowMPS,windowH::Window)
    lenvs = environments(Ψ.left_gs,windowH.left)
    renvs = environments(Ψ.right_gs,windowH.right)
    Window(lenvs, environments(Ψ,windowH.middle;lenvs=lenvs,renvs=renvs), renvs)
end

# we need constructor, agnostic of particular MPS
environments(st,ham::SumOfOperators) = MultipleEnvironments(ham, map(op->environments(st,op),ham.ops) )

environments(st::WindowMPS,ham::SumOfOperators;lenvs=environments(st.left_gs,ham),renvs=environments(st.right_gs,ham)) = 
    MultipleEnvironments(ham, map( (op,sublenv,subrenv)->environments(st,op;lenvs=sublenv,renvs=subrenv),ham.ops,lenvs,renvs) )

environments(st,x::MultipliedOperator) = environments(st,x.op)

