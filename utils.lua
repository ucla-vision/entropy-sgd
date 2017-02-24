local mnist = require 'mnist'
local utils = {}

local function load_mnist()
    local train, test = mnist.traindataset(), mnist.testdataset()
    local tn, vn = 60000, 10000

    train.data = train.data:reshape(tn, 1, 28, 28):float()
    train.data:add(-126):div(126)
    train.label:add(1)

    test.data = test.data:reshape(vn, 1, 28, 28):float()
    test.data:add(-126):div(126)
    test.label:add(1)

    return  {data = train.data, labels=train.label, size=tn},
    {data = test.data, labels=test.label, size=vn},
    {data = test.data, labels=test.label, size=vn}
end

local function load_cifar()
    local n = require 'npy4th'

    local train, test = n.loadnpz('proc/cifar10-train.npz'), n.loadnpz('proc/cifar10-test.npz')

    local d = {data=train.data, labels=train.labels:add(1), size=train.data:size(1)}
    local v = {data=test.data, labels=test.labels:add(1), size=test.data:size(1)}
    return d, v, v
end

function utils.load_dataset()
    if string.find(opt.model, 'mnist') then
        return load_mnist()
    elseif string.find(opt.model, 'cifar') then
        return load_cifar()
    else
        print('Unknown model: ' .. opt.model)
        os.exit(1)
    end
end

function utils.set_gpu()
    torch.setdefaulttensortype('torch.FloatTensor')
    torch.manualSeed(opt.seed)
    torch.setnumthreads(4)
    print('Using GPU')
    require 'cutorch'
    require 'nn'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeedAll(opt.seed)
    require 'cudnn'
    cudnn.benchmark = true
    cudnn.fastest = true
end

return utils