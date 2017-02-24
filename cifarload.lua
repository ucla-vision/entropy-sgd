local n = require 'npy4th'

local dataset = {}
function dataset.split(a,b)
    local train, test

    if opt.cifar100 then
        print('Loading CIFAR-100')
        train = n.loadnpz('/local2/pratikac/cifar/preprocessed/cifar-100-train_all.npz')
        test = n.loadnpz('/local2/pratikac/cifar/preprocessed/cifar-100-test.npz')
    else
        train = n.loadnpz('/local2/pratikac/cifar/preprocessed/train_all.npz')
        test = n.loadnpz('/local2/pratikac/cifar/preprocessed/test.npz')
    end

    if opt.full ~= true then
        print('Overfitting on 10% subset ...')
        local frac = 0.1
        local tn, ten = train.data:size(1), test.data:size(1)

        train.data = train.data:narrow(1,1,frac*tn)
        train.labels = train.labels:narrow(1,1,frac*tn)
        test.data = test.data:narrow(1,1,frac*ten)
        test.labels = test.labels:narrow(1,1,frac*ten)
    end

    local d = {data=train.data, labels=train.labels:add(1), size=train.data:size(1)}
    local v = {data=test.data, labels=test.labels:add(1), size=test.data:size(1)}
    return d, v, v
end

return dataset
