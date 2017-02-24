require 'torch'
local lapp = require 'pl.lapp'

opt = lapp[[
-m,--model          (default 'mnistconv')
-b,--batch_size     (default 128)               Batch size
--LR                (default 0)                 Learning rate
--dropout           (default 0.5)               Dropout
--L                 (default 0)                 Num. Langevin iterations
--gamma             (default 1e-4)              Langevin gamma coefficient
--scoping           (default 1e-3)              Scoping parameter \gamma*(1+scoping)^t
--noise             (default 1e-4)              Langevin dynamics additive noise factor (*stepSize)
-g,--gpu            (default 1)                 GPU id
--L2                (default 0)                 L2 regularization
-s,--seed           (default 42)
-e,--max_epochs     (default 10)
-v,--verbose                                    Print gradient statistics
-h,--help                                       Print this message
]]
print(opt)

if opt.help then
    os.exit()
end

local utils = require 'utils.lua'
local models = require 'models'
require 'entropyoptim'

function trainer(d)
    local x, y = d.data, d.labels
    local w, dw = model:getParameters()
    model:training()
    
    local bs = opt.batch_size
    local num_batches = x:size(1)/bs
    local timer = torch.Timer()
    local timer1 = torch.Timer()

    local loss = 0
    confusion:zero()
    for b =1,num_batches do
        collectgarbage()
        timer1:reset()

        local feval = function(_w, dry)
            local dry = dry or false

            if _w ~= w then w:copy(_w) end
            dw:zero()

            local idx = torch.Tensor(bs):random(1, d.size):type('torch.LongTensor')
            local xc, yc = x:index(1, idx):cuda(), y:index(1, idx):cuda()

            local yh = model:forward(xc)
            local f = cost:forward(yh, yc)
            local dfdy = cost:backward(yh, yc)
            model:backward(xc, dfdy)
            cutorch.synchronize()

            if dry == false then
                loss = loss + f
                confusion:batchAdd(yh, yc)
                confusion:updateValids()
            end
            return f, dw
        end

        optim.entropysgd(feval, w, optim_state)
        if b % 100 == 0 then
            print( ('+[%3d][%3d/%3d] %.5f %.3f%%'):format(epoch,
                b, num_batches, loss/b, (1 - confusion.totalValid)*100))
        end
    end

    loss = loss/num_batches
    print(('Train: [%3d] %.5f %.3f%% [%.2fs]'):format(epoch, loss,
        (1 - confusion.totalValid)*100, timer:time().real))
    print('')
end

function set_dropout(p)
    local p = p or 0
    for i,m in ipairs(model.modules) do
        if m.module_name == 'nn.Dropout' or torch.typename(m) == 'nn.Dropout' then
            m.p = p
        end
    end
    -- set input dropout back
    if opt.model == 'cifarconv' then
        if p > 0 then
            local m = model.modules[1]
            assert(m.module_name == 'nn.Dropout' or torch.typename(m) == 'nn.Dropout')
            m.p = 0.2
        end
    end
end

-- this is a weird hack
-- batch-normalization parameters do not train well due to dropout, so this function sets
-- the dropout to zero, dry-feeds the dataset to let the batch-normalization params settle
-- and then sets the dropout back to its old value again
function compute_bn_params(d)
    set_dropout(0)

    local x, y = d.data, d.labels
    local w, dw = model:getParameters()
    model:training()
    
    local bs = 1024
    local num_batches = math.ceil(x:size(1)/bs)

    for b =1,num_batches do
        collectgarbage()

        local feval = function(_w)
            if _w ~= w then w:copy(_w) end
            dw:zero()

            local sidx,eidx = (b-1)*bs, math.min(b*bs, x:size(1))
            local xc, yc = x:narrow(1, sidx + 1, eidx-sidx):cuda(), y:narrow(1, sidx + 1, eidx-sidx):cuda()

            local yh = model:forward(xc)
            cutorch.synchronize()
            return f, dw
        end
        feval(w)
    end
    set_dropout(opt.dropout)
end

function tester(d)
    compute_bn_params(d)

    local x, y = d.data, d.labels
    model:evaluate()

    local bs = 1024
    local num_batches = math.ceil(x:size(1)/bs)

    local loss = 0
    confusion:zero()
    for b =1,num_batches do
        collectgarbage()

        local sidx,eidx = (b-1)*bs, math.min(b*bs, x:size(1))
        local xc, yc = x:narrow(1, sidx + 1, eidx-sidx):cuda(),
        y:narrow(1, sidx + 1, eidx-sidx):cuda()

        local yh = model:forward(xc)
        local f = cost:forward(yh, yc)
        cutorch.synchronize()

        confusion:batchAdd(yh, yc)
        confusion:updateValids()
        loss = loss + f
        if b % 100 == 0 then
            print( ('*[%2d][%3d/%3d] %.5f %.3f%%'):format(epoch, b, num_batches, loss/b, (1 - confusion.totalValid)*100))
        end
    end
    loss = loss/num_batches
    print( ('Test: [%2d] %.5f %.3f%%'):format(epoch, loss, (1 - confusion.totalValid)*100))
    print('')
end

function learning_rate_schedule()
    local lr = opt.LR
    if opt.LR > 0 then
        print(('[LR] %.5f'):format(lr))
        return lr
    end

    local regimes = {}
    if opt.L == 0 then
        if opt.model == 'mnistfc' or opt.model == 'mnistconv' then
            regimes = {
                {1,30,0.1},
                {30,60,0.1*0.2},
                {60,150,0.1*0.2^2}}
            opt.max_epochs = 100
        elseif opt.model == 'cifarconv' then
            regimes = {
                {1,60, 0.1},
                {60,120, 0.1*0.2^1},
                {120,180, 0.1*0.2^2},
                {180,250, 0.1*0.2^3}}
            opt.max_epochs = 200
        end
    else
        if opt.model == 'mnistfc' then
            regimes = {
                {1,2,1},
                {3,15,0.1}}
            opt.max_epochs = 5
        elseif opt.model == 'mnistconv' then
            regimes = {
                {1,2,1},
                {3,7,0.1},
                {8,15,0.01}}
            opt.max_epochs = 5
        elseif opt.model == 'cifarconv' then
            regimes = {
                {1,3,1},
                {4,6,0.2},
                {7,12,0.04}}
            opt.max_epochs = 10
        end
    end

    for _,row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            lr = row[3]
            break
        end
    end

    print(('[LR] %.5f'):format(lr))
    return lr
end

function main()
    utils.set_gpu()
    model, cost = models.build()
    local train, val, test = utils.load_dataset()

    local classes = torch.totable(torch.range(1,10))
    confusion = optim.ConfusionMatrix(classes)

    optim_state = optim_state or { learningRate= opt.LR,
        learningRateDecay = 0,
        weightDecay = opt.L2,
        momentum = 0.9,
        nesterov = true,
        dampening = 0,
        rho=opt.rho,
        gamma=opt.gamma,
        scoping=opt.scoping,
        L=opt.L,
        noise = opt.noise}

    local freq = 5
    if opt.L > 0 then freq = 1 end
    epoch = epoch or 1
    while epoch <= opt.max_epochs do
        optim_state.learningRate = learning_rate_schedule()
        trainer(train)
        if epoch % freq == 0 then
            tester(val)
        end

        epoch = epoch + 1
        print('')
    end
    print('Finished')
    tester(test)
end

main()
