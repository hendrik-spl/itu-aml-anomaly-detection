import tensorflow as tf

def dssim_loss(y_true, y_pred):
    return 1/2 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))/2

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def ssim_l1_loss(y_true, y_pred, alpha=0.5):
    """
        y_true: Ground truth images.
        y_pred: Predicted images.
        alpha: Weighting factor for SSIM and L1 loss. 
               alpha = 0.5 means equal weight for both losses.
    """    
    # Compute SSIM loss
    ssim = tf.image.ssim(y_true, y_pred, 1.0)
    ssim_loss = 1 - tf.reduce_mean(ssim)
    
    # Compute L1 loss
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # Combine SSIM and L1 losses
    combined_loss = alpha * ssim_loss + (1 - alpha) * l1_loss
    
    return combined_loss