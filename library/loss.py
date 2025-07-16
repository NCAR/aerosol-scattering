import torch
import numpy as np

def NormVarLoss(logvar,mu_delta):
    return torch.mean(0.5*np.log(2*np.pi) + 0.5*logvar + 0.5*mu_delta**2/torch.exp(logvar))

def RegVarEvidentialLoss(pred,mu_delta,reg):
    log_alpha, log_beta = torch.split(pred, pred.shape[-1]//2, dim=-1)
    alpha = torch.exp(log_alpha)+1
    beta = torch.exp(log_beta)
    loss_InvGam = 0.5*np.log(2*np.pi) \
        -alpha*log_beta+(alpha+0.5)*torch.log(mu_delta**2/2+beta) \
        +torch.lgamma(alpha)-torch.lgamma(alpha+0.5) 
    
    reg_loss = torch.abs(mu_delta)*(2+alpha)
    
    return torch.mean(loss_InvGam + reg*reg_loss)

def VarEvidentialLoss(pred,mu_delta):
    log_alpha, log_beta = torch.split(pred, pred.shape[-1]//2, dim=-1)
    alpha = torch.exp(log_alpha)+1
    beta = torch.exp(log_beta)
    loss_InvGam = 0.5*np.log(2*np.pi) \
        -alpha*log_beta+(alpha+0.5)*torch.log(mu_delta**2/2+beta) \
        +torch.lgamma(alpha)-torch.lgamma(alpha+0.5) 
    
    return torch.mean(loss_InvGam)

def EvidentialLoss(pred,y):
    log_alpha, log_beta, gamma_mean, log_v = torch.split(pred, pred.shape[-1]//4, dim=-1)
    alpha = torch.exp(log_alpha)+1
    beta = torch.exp(log_beta)
    v = torch.exp(log_v)
    loss_InvGam = 0.5*torch.log(np.pi/v) \
        -alpha*torch.log(2*beta*(1+v))+(alpha+0.5)*torch.log(v*(y-gamma_mean)**2+2*beta*(1+v)) \
        +torch.lgamma(alpha)-torch.lgamma(alpha+0.5) 
    
    return torch.mean(loss_InvGam)

def RegEvidentialLoss(pred,y,reg):
    log_alpha, log_beta, gamma_mean, log_v = torch.split(pred, pred.shape[-1]//4, dim=-1)
    alpha = torch.exp(log_alpha)+1
    beta = torch.exp(log_beta)
    v = torch.exp(log_v)
    loss_InvGam = 0.5*torch.log(np.pi/v) \
        -alpha*torch.log(2*beta*(1+v))+(alpha+0.5)*torch.log(v*(y-gamma_mean)**2+2*beta*(1+v)) \
        +torch.lgamma(alpha)-torch.lgamma(alpha+0.5) 
    
    reg_loss = torch.abs(y-gamma_mean)*(2*v+alpha)
    
    return torch.mean(loss_InvGam + reg*reg_loss)