import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
import numpy as np
import torch
import cv2   # هنا هنستخدم OpenCV

# ------------------------------
# Utilities
# ------------------------------
def dark_channel(img, ps=15):
    H, W, C = img.shape
    pad = ps // 2
    img_padded = np.pad(img, [(pad, pad), (pad, pad), (0, 0)], mode='edge')
    dark = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            patch = img_padded[i:i+ps, j:j+ps, :]
            dark[i,j] = np.min(patch)
    return dark

def atm_light(img, dark, px=1e-3):
    flat_img = img.reshape(-1, img.shape[2])
    flat_dark = dark.flatten()
    numpx = max(1, int(flat_dark.size * px))
    indices = np.argsort(-flat_dark)[:numpx]
    A = np.mean(flat_img[indices], axis=0)
    return A

def transmission_est(img, A, w=0.95, ps=15):
    norm_img = img / A
    dark_norm = dark_channel(norm_img, ps)
    t = 1 - w * dark_norm + 0.25
    return t

def boxfilter(m, r):
    H, W = m.shape
    ysum = np.cumsum(m, axis=0)
    mp = np.zeros_like(m)
    mp[:r+1,:] = ysum[r:2*r+1,:]
    mp[r+1:H-r,:] = ysum[2*r+1:,:] - ysum[:H-2*r-1,:]
    mp[H-r:,:] = np.tile(ysum[-1,:], (r,1)) - ysum[H-2*r-1:H-r-1,:]
    xsum = np.cumsum(mp, axis=1)
    mp[:,:r+1] = xsum[:,r:2*r+1]
    mp[:,r+1:W-r] = xsum[:,2*r+1:] - xsum[:,:W-2*r-1]
    mp[:,-r:] = np.tile(xsum[:,-1][:,None], (1,r)) - xsum[:,W-2*r-1:W-r-1]
    return mp

def guided_filter(I, p, r=40, eps=1e-3):
    H, W, C = I.shape
    S = boxfilter(np.ones((H,W)), r)
    mean_I = np.zeros((C,H,W))
    mean_ip = np.zeros((C,H,W))
    cov_ip = np.zeros((C,H,W))
    for c in range(C):
        mean_I[c] = boxfilter(I[:,:,c], r)/S
        mean_ip[c] = boxfilter(I[:,:,c]*p, r)/S
        cov_ip[c] = mean_ip[c] - mean_I[c]*boxfilter(p,r)/S
    var_I = np.zeros((C,C,H,W))
    var_I[0,0] = boxfilter(I[:,:,0]*I[:,:,0], r)/S - mean_I[0]*mean_I[0]
    var_I[0,1] = boxfilter(I[:,:,0]*I[:,:,1], r)/S - mean_I[0]*mean_I[1]
    var_I[0,2] = boxfilter(I[:,:,0]*I[:,:,2], r)/S - mean_I[0]*mean_I[2]
    var_I[1,1] = boxfilter(I[:,:,1]*I[:,:,1], r)/S - mean_I[1]*mean_I[1]
    var_I[1,2] = boxfilter(I[:,:,1]*I[:,:,2], r)/S - mean_I[1]*mean_I[2]
    var_I[2,2] = boxfilter(I[:,:,2]*I[:,:,2], r)/S - mean_I[2]*mean_I[2]
    a = np.zeros((H,W,C))
    for i in range(H):
        for j in range(W):
            sigma = np.array([[var_I[0,0,i,j], var_I[0,1,i,j], var_I[0,2,i,j]],
                              [var_I[0,1,i,j], var_I[1,1,i,j], var_I[1,2,i,j]],
                              [var_I[0,2,i,j], var_I[1,2,i,j], var_I[2,2,i,j]]])
            cov_ij = np.array([cov_ip[0,i,j], cov_ip[1,i,j], cov_ip[2,i,j]])
            a[i,j] = cov_ij @ np.linalg.inv(sigma + eps*np.eye(3))
    b = boxfilter(a[:,:,0],r)*I[:,:,0] + boxfilter(a[:,:,1],r)*I[:,:,1] + boxfilter(a[:,:,2],r)*I[:,:,2] + boxfilter(boxfilter(p,r)/S - a[:,:,0]*mean_I[0] - a[:,:,1]*mean_I[1] - a[:,:,2]*mean_I[2],r)
    return b/S

def recover(img, A, t, tmin=0.1):
    J = np.zeros_like(img)
    for c in range(img.shape[2]):
        J[:,:,c] = (img[:,:,c]-A[c])/np.maximum(t,tmin)+A[c]
    return J/np.max(J)
# ------------------------------
# Function: Enhancement (Saturation + Contrast)
# ------------------------------
def enhance_image(pil_img, sat=2.0, contrast=3.0):
    img = ImageEnhance.Color(pil_img).enhance(sat)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img

# ------------------------------
# Function: Compute MSE between PIL or numpy image and original tensor
# ------------------------------
def compute_mse(img_input, orig_tensor):
    if isinstance(img_input, Image.Image):
        tensor_input = torch.tensor(np.array(img_input)/255.).permute(2,0,1).unsqueeze(0).to(orig_tensor.device)
    elif isinstance(img_input, np.ndarray):
        tensor_input = torch.tensor(img_input).permute(2,0,1).unsqueeze(0).to(orig_tensor.device)
    else:
        tensor_input = img_input
    return F.mse_loss(tensor_input, orig_tensor).item()

# ------------------------------
# Function: Dehaze
# ------------------------------
def dehaze_image(np_img_float):
    jdark = dark_channel(np_img_float)
    A = atm_light(np_img_float, jdark)
    t_raw = transmission_est(np_img_float, A)
    J = recover(np_img_float, A, t_raw)
    return J



# ---- Unsharp Masking Function ----
def unsharp_mask(img_np, kernel_size=(5,5), sigma=1.0, amount=1.5, threshold=0):
    """
    img_np: صورة numpy array بقيم 0-1
    kernel_size, sigma: بارامترات Gaussian blur
    amount: شدة الحدة
    threshold: لتجاهل الفرق الصغير (noise suppression)
    """
    # نحول للصيغة 0-255
    img_uint8 = (img_np*255).astype(np.uint8)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(img_uint8, kernel_size, sigma)
    
    # Sharpen
    sharpened = float(amount + 1) * img_uint8 - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255*np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)

    # threshold لو محتاجه
    if threshold > 0:
        low_contrast_mask = np.abs(img_uint8 - blurred) < threshold
        np.copyto(sharpened, img_uint8, where=low_contrast_mask)

    return sharpened / 255.0


import cv2
import numpy as np

def smooth_image(img_np, method="bilateral", kernel_size=5):
    """
    img_np: صورة numpy 0-1 (float)
    method: "gaussian" | "median" | "bilateral"
    kernel_size: حجم الفلتر
    """
    img_uint8 = (img_np * 255).astype(np.uint8)

    if method == "gaussian":
        smoothed = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), 0)

    elif method == "median":
        smoothed = cv2.medianBlur(img_uint8, kernel_size)

    elif method == "bilateral":
        smoothed = cv2.bilateralFilter(img_uint8, d=kernel_size, sigmaColor=75, sigmaSpace=75)

    else:
        raise ValueError("Unknown method. Use: gaussian | median | bilateral")

    return smoothed / 255.0