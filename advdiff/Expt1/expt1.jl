#### Expt 1
# Baseline case: numerics are same between ROM and GT
using DifferentialEquations,SciMLSensitivity,Zygote,LinearAlgebra, NPZ, Plots, FileIO, JLD2
using SparseArrays, Random, Statistics, Optimisers, Flux, ForwardDiff, Calculus

begin 
    global N = 100                      # number of cells 
    global L = 2π                       # total length of 1d sim
    global T = 0.5                       # total time 
    global nt = 50                # number of timesteps
    global dt = T / nt                     # time step
    global dx = L/N                      # spatial step
    global x = 0.0 : dx : (L-dx)         # discretized spatial dimension 
    global xgrid = collect(x)            # grid 
    global tsteps = 0.0:dt:T             # discretized time dimension 
    global tspan = (0,T)                 # end points of time integration for ODEProbem
    global train_nt = 3       # only applies to autoregressive learning. The number of timesteps to train on, even if inference is performed for full "nt". For reasons of stability, and is a hyper-parameter
    global t = collect(tsteps)
    global ν = 3e-02                       # viscosity
    global ckpt_save_freq = 20
    global epochs = 1000
    global lr = 0.01
end 

#include("/Users/arvindm/Documents/PROJECTS/githubRepos/diffprog_NIROM_KdV/numericalmethods_sensitivity/src/numerical_derivatives.jl");
include("/vast/home/arvindm/PROJECTS/diffprog_NIROM_KdV/numericalmethods_sensitivity/src/numerical_derivatives.jl");

checkpoint_fpath = "./ckpt_expt1.jld2"

resume_checkpoint = false

# generate training data
∂x1 = f1_firstOrder_backward(N,dx) 
∂x2 = f2_sixthOrder_central(N,dx)

#ROM numerical scheme 
∂x1_rom = f1_firstOrder_backward(N,dx) 

function ic_definition(N)
    u0 = ones(N)
    u0[2:Int(N/2)] .= 2.0
    return u0
end
u0 = ic_definition(N)

################################################################################
#### Training Data Definition
################################################################################
pones = ones(N); 

function true_advdiff(u,p,t)
    u = - u .* (∂x1*u) + ν .* (∂x2*u) .*  p
    return u
end

function solve_true_advdiff(p,u,tspan)
    prob = ODEProblem(true_advdiff, u, tspan,p)
    solve(prob,RK4(),alg_hints=[:stiff], saveat=dt)
end

true_sol = Array(solve_true_advdiff(pones,u0,tspan))

#### Model and NeuralPDE Definition

model = Chain(
            Dense(N,20,tanh),
            Dense(20,20,tanh),
            Dense(20,N));


if resume_checkpoint == true
    f = jldopen(checkpoint_fpath,"r")
    p = f["p"];
    optim = f["state"];
    close(f)
    println("loaded checkpoint to RESUME training...")
    _, re = Flux.destructure(model);
    model = re(p); #load trained weights from checkpoint into model state.
elseif resume_checkpoint == false  
    println("Starting training from SCRATCH!!!")
    p, re = Flux.destructure(model); #extract random init weights from model for training.
    optim = Flux.setup(Flux.Adam(lr), p);
else
    println("UNKNOWN training state, assuming training from scratch")
    p, re = Flux.destructure(model); #extract random init weights from model for training.
    optim = Flux.setup(Flux.Adam(lr), p);
end

function learned_advdiff(u,p,t)
    ϕ = re(p)
    u = - u .* (∂x1_rom*u) + ϕ(u)
    return u
end 

function solve_learned_advdiff(p,u,tspan_step) 
    """ tspan_step indicates cyclic prediction with small tspan over longer time windows """
    prob = ODEProblem(learned_advdiff, u, tspan_step,p)
    solve(prob,RK4(),alg_hints=[:stiff], saveat=dt)
end

################################################################################
### Telemetry for Training Autoregressive solution minimizing every timestep
################################################################################

# Training loop over multiple epochs
optim = Flux.setup(Flux.Adam(lr), p);  # will store optimiser momentum, etc.
@show train_nt

T_step = dt
tsteps_step = 0.0:dt:T_step  
tspan_step = (0,T_step)
losses = []

for epoch in 1:epochs
    @show epoch
    du0_true_training = []
    du0_pred_training = []
    dp_pred_training = []
    pred_sol_training = []
    true_sol_training = []
    p_training = []
    nnpred_training = []
    
    u0_local = copy(u0);
    sol_autoregressive = zeros(N,train_nt)
    sol_true_advdiff_full = zeros(N,train_nt)
    sol_true_advdiff_full[:,1] = u0;
    for t=2:train_nt
        #Step 0: ut is solution at previous iteration! must be updated
        ut_true =  Array(solve_true_advdiff(pones,u0_local,tspan_step)[end]);
        sol_true_advdiff_full[:,t] = copy(ut_true);
        #solve eqn. again to compute jacobian
        _,du0_true = Zygote.jacobian((p,u) -> solve_true_advdiff(p,u,tspan_step)[end],pones,u0_local);
        
        # Step 1: Compute gradient for backprop using true solution and true adjoint
        loss, ∇p = Flux.withgradient(p) do p
            ut = Array(solve_learned_advdiff(p,u0_local,tspan_step)[end])
            loss = Flux.mse(ut, true_sol[:,t])
            @show loss
            return loss
        end
        #compute predicted solution again to calculate jacobian with AD. This is inefficient.. but works for now.
        ut_again  = Array(solve_learned_advdiff(p,u0_local,tspan_step)[end])
        nnpred = re(p)(u0_local)
        dp_pred,du0_pred = Zygote.jacobian((p,u) -> solve_learned_advdiff(p,u,tspan_step)[end],p,u0_local);

        #save telemetry
        push!(true_sol_training, ut_true)
        push!(pred_sol_training,ut_again)
        push!(du0_true_training,du0_true);        
        push!(du0_pred_training,du0_pred)
        push!(dp_pred_training,dp_pred)
        push!(p_training,p)
        push!(nnpred_training,nnpred)

        # Step 2: Optimizer update
        Flux.update!(optim, p, ∇p[1])
        push!(losses,loss)

        if epoch%ckpt_save_freq == 0
            jldsave(checkpoint_fpath,p=p,gradp=∇p,state=optim)
            println("saved checkpoint")
        elseif epoch == epochs # to save final trained state.
            jldsave(checkpoint_fpath,p=p,gradp=∇p,state=optim)
            println("saved FINAL checkpoint")
        end        

        sol_autoregressive[:,t] = copy(ut_again)
        u0_local = ut_again;
    end

    #stack all lists into arrays so its writable/readable with NPZ to Python
    true_sol_training = stack(true_sol_training; dims=2);
    pred_sol_training = stack(pred_sol_training; dims=2);
    du0_true_training = stack(du0_true_training; dims=3);
    du0_pred_training = stack(du0_pred_training; dims=3);
    dp_pred_training = stack(dp_pred_training; dims=3);
    p_training = stack(p_training; dims=2);
    nnpred_training = stack(nnpred_training; dims=2);

    if epoch%ckpt_save_freq == 0
        println("save telemetry")
        fname = "telemetry_caseA_expt1_epoch" * string(epoch)  * ".npz"
        varsdict = Dict("p" => p,"true_sol_training" => true_sol_training,"du0_true_training" => du0_true_training, "pred_sol_training" => pred_sol_training, "du0_pred_training" => du0_pred_training, "dp_pred_training" => dp_pred_training, "p_training" => p_training, "nnpred_training" => nnpred_training)
        npzwrite(fname, varsdict)
    end
end

loss_history = stack(losses;dims=1)
println("Completed training, writing loss history...")
npzwrite("trainloss_caseA_expt1.npz", Dict("loss_history" => loss_history))

################################################################################
### Inference Telemetry on Train u0
################################################################################


nt_inference = nt
tspan_inference = (0,nt_inference*dt)
sol_true_advdiff_full = Array(solve_true_advdiff(pones,u0,tspan_inference))

sol_pred_advdiff_full = zeros(N,nt_inference)
du0_pred_inference_train = []
dp_pred_inference_train = []
global u0_local = u0;
for t=2:nt_inference
    @show t
    T_step = dt
    tsteps_step = 0.0:dt:T_step  
    tspan_step = (0,T_step)
    u_pred = Array(solve_learned_advdiff(p,u0_local,tspan_step)[end])
    sol_pred_advdiff_full[:,t] = copy(u_pred);
    global u0_local = u_pred;
    dp_pred,du0_pred = Zygote.jacobian((p,u) -> solve_learned_advdiff(p,u,tspan_step)[end],p,u0_local);
    push!(du0_pred_inference_train,du0_pred)
    push!(dp_pred_inference_train,dp_pred)
end
du0_pred_inference_train = stack(du0_pred_inference_train; dims=3);
dp_pred_inference_train = stack(dp_pred_inference_train; dims=3);
fname = "telemetry_caseA_expt1_inference_train.npz"
varsdict = Dict("du0_pred_inference_train" => du0_pred_inference_train, "dp_pred_inference_train" => dp_pred_inference_train, "sol_true_advdiff_full" => sol_true_advdiff_full)
npzwrite(fname,varsdict)

################################################################################
### Inference Telemetry with unseen u0
################################################################################

nt_inference = nt
tspan_inference = (0,nt_inference*dt)
sol_true_advdiff_full = Array(solve_true_advdiff(pones,u0,tspan_inference))

sol_pred_advdiff_full = zeros(N,nt_inference)
du0_pred_inference_test = []
dp_pred_inference_test = []

function ic_definition_test(N,amp)
    u0 = ones(N)
    u0[2:Int(N/2)] .= amp
    return u0
end
u0 = ic_definition_test(N,10.0)

global u0_local = u0;
for t=2:nt_inference
    @show t
    T_step = dt
    tsteps_step = 0.0:dt:T_step  
    tspan_step = (0,T_step)
    u_pred = Array(solve_learned_advdiff(p,u0_local,tspan_step)[end])
    sol_pred_advdiff_full[:,t] = copy(u_pred);
    global u0_local = u_pred;
    dp_pred_test,du0_pred_test = Zygote.jacobian((p,u) -> solve_learned_advdiff(p,u,tspan_step)[end],p,u0_local);
    push!(du0_pred_inference_test,du0_pred_test)
    push!(dp_pred_inference_test,dp_pred_test)
end
du0_pred_inference_test = stack(du0_pred_inference_test; dims=3);
dp_pred_inference_test = stack(dp_pred_inference_test; dims=3);
fname = "telemetry_caseA_expt1_inference_test.npz"
varsdict = Dict("du0_pred_inference_test" => du0_pred_inference_test, "dp_pred_inference_test" => dp_pred_inference_test, "sol_true_advdiff_full" => sol_true_advdiff_full)
npzwrite(fname,varsdict)
println("DONE.")
