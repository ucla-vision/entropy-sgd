local models = {}

local function mnistconv()
    local m = nn:Sequential()
    opt.dropout = 0.5

    local c1, c2, c3 = 20, 50, 500
    m:add(cudnn.SpatialConvolution(1, c1, 5, 5))
    m:add(cudnn.ReLU())
    m:add(cudnn.SpatialMaxPooling(3,3,3,3))
    m:add(cudnn.SpatialBatchNormalization(c1))
    m:add(nn.Dropout(opt.dropout))

    m:add(cudnn.SpatialConvolution(c1, c2, 5, 5))
    m:add(cudnn.ReLU())
    m:add(cudnn.SpatialMaxPooling(2,2,2,2))
    m:add(cudnn.SpatialBatchNormalization(c2))
    m:add(nn.Dropout(opt.dropout))

    m:add(nn.View(c2*2*2))
    m:add(nn.Linear(c2*2*2, c3))
    m:add(cudnn.ReLU())
    m:add(nn.Dropout(opt.dropout))

    m:add(nn.Linear(c3, 10))
    m:add(cudnn.LogSoftMax())

    return m
end

local function mnistfc()
    local m = nn:Sequential():add(nn.View(784))
    local c = 1024
    local p = 2
    opt.dropout = 0.5

    local c1 = c
    for i=1,p do
        if i == 1 then c1 = 784 else c1 = c end
        m:add(nn.Linear(c1, c))
        m:add(nn.ReLU(true))
        m:add(nn.Dropout(opt.dropout))
    end

    m:add(nn.Linear(c, 10))
    m:add(cudnn.LogSoftMax())
    return m
end

local function cifarconv()
    local c1, c2 = 96, 192
    opt.dropout = 0.5

    -- ALL-CNN-C
    local function convbn(...)
        local arg = {...}
        return nn.Sequential()
        :add(cudnn.SpatialConvolution(...))
        :add(cudnn.SpatialBatchNormalization(arg[2]))
        :add(cudnn.ReLU(true))
    end

    local m = nn.Sequential()
    :add(nn.Dropout(0.2))
    :add(convbn(3,c1,3,3,1,1,1,1))
    :add(convbn(c1,c1,3,3,1,1,1,1))
    :add(convbn(c1,c1,3,3,2,2,1,1))
    :add(nn.Dropout(opt.dropout))
    :add(convbn(c1,c2,3,3,1,1,1,1))
    :add(convbn(c2,c2,3,3,1,1,1,1))
    :add(convbn(c2,c2,3,3,2,2,1,1))
    :add(nn.Dropout(opt.dropout))
    :add(convbn(c2,c2,3,3,1,1,1,1))
    :add(convbn(c2,c2,3,3,1,1,1,1))
    :add(convbn(c2,10,1,1,1,1))
    :add(nn.SpatialAveragePooling(8,8))
    :add(nn.View(10))
    :add(cudnn.LogSoftMax())

    return m
end

function models.build()
    local m
    if opt.model == 'mnistfc' then
        m = mnistfc()
    elseif opt.model == 'mnistconv' then
        m = mnistconv()
    elseif opt.model == 'cifarconv' then
        m = cifarconv()
    else
        assert('Unknown opt.model: ' .. opt.model)
    end

    local cost = nn.ClassNLLCriterion()
    return m:cuda(), cost:cuda()
end

return models