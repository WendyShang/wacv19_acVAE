-- generalized KLD between two multivariate Gaussian distribution
-- (mean1, logvar1) || (mean2, logvar2)

local GKLDCriterion, parent = torch.class('nn.GKLDCriterion', 'nn.Criterion')

function GKLDCriterion:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
end

function GKLDCriterion:updateOutput(info1, info2)
    -- Conditional VAE paper:
    -- cost = -0.5*(1+log_var1) + 0.5*log_var2 + 0.5*(var1 + (mu1-mu2)^2)/var2
    -- special case: mean2 = 0, log_var2 = 0
    -- cost = -0.5*(1+log_var1) + 0.5*(var1 + mu1^2)
    local mean1, log_var1 = table.unpack(info1)
    local mean2, log_var2 = table.unpack(info2)
    local mean_diff_sq = torch.pow(mean1-mean2, 2)
    local var1 = torch.exp(log_var1)
    local var2 = torch.cmax(torch.exp(log_var2), 0.00001)
    local KLDelements = -0.5*(1+log_var1) + 0.5*log_var2 + 0.5*torch.cdiv(var1 + mean_diff_sq, var2)
    local output = torch.sum(KLDelements)
    if self.sizeAverage then
        output = output / mean1:size(1)
    end
    self.output = output
    return self.output
end

function GKLDCriterion:updateGradInput(info1, info2)
    -- Conditional VAE paper:
    -- dmu1 = (mu1-mu2)/var2
    -- dmu2 = -(mu1-mu2)/var2 = -dmu2
    -- dlog_var1 = -0.5*(1-var1/var2)
    -- dlog_var2 = 0.5*(1-(var1+(mu1-mu2)^2)/var2)
    -- special case: mean2 = 0, log_var2 = 0
    -- dmu1 = 0.5*mu1
    -- dlog_var1 = -0.5*(1-var1)
    local mean1, log_var1 = table.unpack(info1)
    local mean2, log_var2 = table.unpack(info2)
    local mean_diff = (mean1-mean2)
    local mean_diff_sq = torch.pow(mean_diff, 2)
    local var1 = torch.exp(log_var1)
    local var2 = torch.cmax(torch.exp(log_var2), 0.00001)

    self.gradInput = {}

    self.gradInput[1] = mean_diff:clone()
    self.gradInput[1]:cdiv(var2)
    self.gradInput[3] = self.gradInput[1]:clone()
    self.gradInput[3]:mul(-1)

    self.gradInput[2] = torch.cdiv(var1, var2)
    self.gradInput[2]:mul(-1):add(1):mul(-0.5)
    self.gradInput[4] = torch.cdiv(var1 + mean_diff_sq, var2)
    self.gradInput[4]:mul(-1):add(1):mul(0.5)

    if self.sizeAverage then
        self.gradInput[1] = self.gradInput[1]/(mean1:size(1))
        self.gradInput[2] = self.gradInput[2]/(log_var1:size(1))
        self.gradInput[3] = self.gradInput[3]/(mean2:size(1))
        self.gradInput[4] = self.gradInput[4]/(log_var2:size(1))
    end
    return self.gradInput
end
