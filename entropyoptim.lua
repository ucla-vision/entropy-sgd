local cjson = require 'cjson'
require 'optim'

function optim.entropysgd(opfunc, x, config, state)
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 1e-3
    local lrd = config.learningRateDecay or 0
    local wd = config.weightDecay or 0
    local mom = config.momentum or 0
    local damp = config.dampening or mom
    local nesterov = config.nesterov or false

    -- averaging parameters
    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    local gamma = config.gamma or 0
    local scoping = config.scoping or 0
    local noise = config.noise or 1e-3
    state.sgld = state.sgld or {beta1=0.75}
    local sgld = state.sgld

    state.t = state.t or 0
    local nevals = state.t
    assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

    -- (1) evaluate f(x) and df/dx
    local xc = x:clone()
    local fx,dfdx = opfunc(x, false)
    state.t = state.t + 1

    -- (2) weight decay with single or individual parameters
    if wd ~= 0 then
        dfdx:add(wd, x)
    end

    -- (3) apply momentum
    if mom ~= 0 then
        if not state.dfdx then
            state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
        else
            state.dfdx:mul(mom):add(1-damp, dfdx)
        end
        if nesterov then
            dfdx:add(mom, state.dfdx)
        else
            dfdx = state.dfdx
        end
    end

    -- (4) learning rate decay (usual annealing)
    local clr = lr / (1 + nevals*lrd)

    -- (x-<x>) that is added from Langevin
    sgld.lx = xc:clone()
    sgld.lmx = sgld.lx:clone()
    sgld.mdfdx = sgld.mdfdx or xc:clone():zero()
    sgld.xxpd = 0

    sgld.g = gamma*(1+scoping)^state.t

    sgld.eta = sgld.eta or x.new(dfdx:size()):zero()
    sgld.w = sgld.w or x.new(dfdx:size()):zero()
    sgld.w:zero()

    local lclr = config.lclr or 0.1

    if config.L > 0 then
        local lx = sgld.lx
        local lmx = sgld.lmx
        local eta = sgld.eta
        local mdfdx = sgld.mdfdx:zero()
        local g = sgld.g

        local debug_states = {}
        for i=1,config.L do
            local lfx,ldfdx = opfunc(lx, true)

            if mom ~= 0 then
                mdfdx:mul(mom):add(1-damp, ldfdx)
            end
            if nesterov then
                ldfdx:add(mom, mdfdx)
            else
                ldfdx = mdfdx
            end

            eta:normal()
            ldfdx:add(-g, xc-lx):add(wd,lx):add(noise/math.sqrt(0.5*lclr), eta)

            -- update and average
            lx:add(-lclr, ldfdx)

            lmx:mul(sgld.beta1):add(1-sgld.beta1, lx)

            -- collect statistics
            sgld.xxpd = sgld.xxpd + torch.norm(xc-lx)
        end
        sgld.xxpd = sgld.xxpd/config.L
        sgld.w:copy(xc-lmx)
    end

    if opt.verbose and state.t % 50 == 1 then
        local debug_stats = {df=torch.norm(dfdx),
        dF=torch.norm(sgld.w),
        dfdF = torch.dot(dfdx/torch.norm(dfdx), sgld.w/(torch.norm(sgld.w)+1e-6)),
        eta = torch.norm(sgld.eta*noise/math.sqrt(0.5*lclr)),
        xxpd = sgld.xxpd,
        g = sgld.g}
        print(cjson.encode(debug_stats))

        sgld.dfdF = debug_stats.dfdF
    end

    if opt.L > 0 then
        -- kill the original gradient if we are using Entropy-SGD
        dfdx:mul(0)
    end

    x:copy(xc)
    dfdx:add(sgld.w)

    x:add(-clr, dfdx)

    return x,{fx}
end
