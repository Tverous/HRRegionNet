import torch

def calc_loss_all(pred, gt, metrics):
    
#     print('pred = ', pred.shape)
#     print('gt = ', gt.shape)
    
    mask = torch.sign(gt[..., 1])
    N = torch.sum(mask)
    
    if N == 0:
        N = 1
        
    _heatmap_loss = heatmap_loss(pred, gt, mask)
    _size_loss = size_loss(pred, gt, mask)
    _offset_loss = offset_loss(pred, gt, mask) 
    _region_loss = region_loss(pred, gt, mask)
    
    all_loss = (-1 * _heatmap_loss + 10. * _size_loss + 5. * _offset_loss + 5 * _region_loss) / N 
    
    metrics['loss'] = all_loss.item() 
    metrics['heatmap'] = (-1 *  _heatmap_loss / N).item()
    metrics['size'] = (10. * _size_loss / N).item()
    metrics['offset'] = (5. * _offset_loss / N).item()
    metrics['region'] = (5 *  _region_loss / N).item()
    
    return all_loss

def calc_loss_heatmap_only(pred, gt, metrics):
    mask = torch.sign(gt[..., 1])
    N = torch.sum(mask)
    
    if N == 0:
        N = 1
        
    _heatmap_loss = heatmap_loss(pred, gt, mask)
    
    all_loss = (-1 * _heatmap_loss) / N 
    
    metrics['loss'] = all_loss.item() 
    metrics['heatmap'] = (-1 *  _heatmap_loss / N).item()
    
    return all_loss


def heatmap_loss(pred, gt, mask):
    
    alpha = 2.
    beta = 4.
    
    heatmap_gt_rate = torch.flatten(gt[...,0])
    heatmap_gt = torch.flatten(gt[..., 1])
    heatmap_pred = torch.flatten(pred[:, 0,...])
    
    heatloss = torch.sum(heatmap_gt * ((1 - heatmap_pred) ** alpha) * torch.log(heatmap_pred + 1e-9) + 
              (1 - heatmap_gt) * ((1 - heatmap_gt_rate) ** beta) * (heatmap_pred ** alpha) * torch.log(1 - heatmap_pred + 1e-9))
    
    return heatloss

def offset_loss(pred, gt, mask):
    
    offsetloss = torch.sum(torch.abs(gt[...,2] - pred[:,1,...]*mask) + torch.abs(gt[...,3] - pred[:,2, ...] * mask))
    
    return offsetloss

def size_loss(pred, gt, mask):
    
    sizeloss = torch.sum(torch.abs(gt[...,4] - pred[:,3, ...]*mask) + torch.abs(gt[...,5] - pred[:,4,...] * mask))
    
    return sizeloss

def region_loss(pred, gt, mask):
        
    regionloss = torch.sum(torch.log(torch.abs(mask - (((pred[:, 0, ...] + 1e-9) * pred[:, 3, ...] * pred[:, 4, ...] * mask) / (gt[..., 1] * gt[..., 4] * gt[..., 5] + 1e-9))) + 1))
    
    return regionloss


