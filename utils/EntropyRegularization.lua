require 'nn'

local EntropyRegularization, parent = torch.class('nn.EntropyRegularization', 'nn.Criterion')

function EntropyRegularization:__init(sizeAverage)
    parent.__init(self)
    --self.sm = nn.SoftMax()
    self.eps = 1e-7
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
end

function EntropyRegularization:updateOutput(input)
    -- input: B x C
    --local sm = self.sm:updateOutput(input):clamp(self.eps, 1-self.eps)
    input:clamp(self.eps, 1-self.eps)
    self.loss = torch.cmul(input, torch.log(input)):sum(2)
    self.output = -self.loss:sum()
    if self.sizeAverage then
        self.output = self.output / (input:nElement()/input:size(2))
    end
    return self.output
end

function EntropyRegularization:updateGradInput(input)
    --local sm = self.sm.output
    self.gradInput = torch.log(input):add(1):mul(-1)
    --self.gradInput = torch.cmul(sm, torch.expandAs(self.loss, sm) - torch.log(sm))
    if self.sizeAverage then
        self.gradInput = self.gradInput:mul(1/(input:nElement()/input:size(2)))
    end
    return self.gradInput
end