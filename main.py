import sys
import os
import math
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, pack, unpack
from PIL import Image
try:
    import pillow_avif
except ImportError:
    pass
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# Utils & Modules
# -----------------------------------------------------------------------------

def swish(x):
    return x*torch.sigmoid(x)

def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    if x.dim() < 3:
        raise ValueError("Expecting a channels-first (*CHW) tensor of at least 3 dimensions")
    c, h, w = x.shape[-3:]
    s = block_size**2
    if c % s != 0:
        raise ValueError(f"Expecting a channels-first (*CHW) tensor with C divisible by {s}, but got C={c} channels")
    outer_dims = x.shape[:-3]
    x = x.view(-1, block_size, block_size, c // s, h, w)
    x = x.permute(0, 3, 4, 1, 5, 2)
    x = x.contiguous().view(*outer_dims, c // s, h * block_size, w * block_size)
    return x

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, use_conv_shortcut=False, use_agn=False):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.use_conv_shortcut = use_conv_shortcut
        self.use_agn = use_agn

        if not use_agn:
            self.norm1 = nn.GroupNorm(32, in_filters, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, out_filters, eps=1e-6)

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)

        if in_filters != out_filters:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1, bias=False)
            else:
                self.nin_shortcut = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x, **kwargs):
        residual = x
        if not self.use_agn:
            x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_filters != self.out_filters:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)
        return x + residual

class Upsampler(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim * 4
        self.conv1 = nn.Conv2d(dim, dim_out, (3, 3), padding=1)
        self.depth2space = depth_to_space

    def forward(self, x):
        out = self.conv1(x)
        out = self.depth2space(out, block_size=2)
        return out

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, z_channel, in_filters, num_groups=32, eps=1e-6):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=in_filters, eps=eps, affine=False)
        self.gamma = nn.Linear(z_channel, in_filters)
        self.beta = nn.Linear(z_channel, in_filters)
        self.eps = eps
    
    def forward(self, x, quantizer):
        B, C, _, _ = x.shape
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps 
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        bias = rearrange(quantizer, "b c h w -> b c (h w)")
        bias = bias.mean(dim=-1)
        bias = self.beta(bias).view(B, C, 1, 1)
       
        x = self.gn(x)
        x = scale * x + bias
        return x

class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False):
        super().__init__()
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(ch_mult)
        
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=(3, 3), padding=1, bias=False)

        self.down = nn.ModuleList()
        in_ch_mult = (1,)+tuple(ch_mult)
        for i_level in range(self.num_blocks):
            block = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level < self.num_blocks - 1:
                down.downsample = nn.Conv2d(block_out, block_out, kernel_size=(3, 3), stride=(2, 2), padding=1)
            self.down.append(down)
        
        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        self.norm_out = nn.GroupNorm(32, block_out, eps=1e-6)
        self.conv_out = nn.Conv2d(block_out, z_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_in(x)
        for i_level in range(self.num_blocks):
            for i_block in range(self.num_res_blocks):
                x = self.down[i_level].block[i_block](x)
            if i_level < self.num_blocks - 1:
                x = self.down[i_level].downsample(x)
        
        for res in range(self.num_res_blocks):
            x = self.mid_block[res](x)
        
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, in_channels, num_res_blocks, z_channels, ch_mult=(1, 2, 2, 4), 
                resolution, double_z=False):
        super().__init__()
        self.ch = ch
        self.num_blocks = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        block_in = ch*ch_mult[self.num_blocks-1]

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=(3, 3), padding=1, bias=True)

        self.mid_block = nn.ModuleList()
        for res_idx in range(self.num_res_blocks):
            self.mid_block.append(ResBlock(block_in, block_in))
        
        self.up = nn.ModuleList()
        self.adaptive = nn.ModuleList()

        for i_level in reversed(range(self.num_blocks)):
            block = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            self.adaptive.insert(0, AdaptiveGroupNorm(z_channels, block_in))
            for i_block in range(self.num_res_blocks):
                block.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            up = nn.Module()
            up.block = block
            if i_level > 0:
                up.upsample = Upsampler(block_in)
            self.up.insert(0, up)
        
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=(3, 3), padding=1)
    
    def forward(self, z):
        style = z.clone()
        z = self.conv_in(z)
        for res in range(self.num_res_blocks):
            z = self.mid_block[res](z)
        
        for i_level in reversed(range(self.num_blocks)):
            z = self.adaptive[i_level](z, style)
            for i_block in range(self.num_res_blocks):
                z = self.up[i_level].block[i_block](z)
            if i_level > 0:
                z = self.up[i_level].upsample(z)
        
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)
        return z

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

class LFQ(nn.Module):
    def __init__(self, *, dim=None, codebook_size=None, num_codebooks=1, token_factorization=False, factorized_bits=[9,9], **kwargs):
        super().__init__()
        assert exists(dim) or exists(codebook_size)
        self.codebook_size = default(codebook_size, lambda: 2 ** dim)
        self.codebook_dim = int(math.log2(codebook_size))
        codebook_dims = self.codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.dim = dim
        self.num_codebooks = num_codebooks
        self.token_factorization = token_factorization
        
        if not self.token_factorization:
            self.register_buffer('mask', 2 ** torch.arange(self.codebook_dim), persistent=False)
        else:
            self.factorized_bits = factorized_bits
            self.register_buffer("pre_mask", 2** torch.arange(factorized_bits[0]), persistent=False)
            self.register_buffer("post_mask", 2**torch.arange(factorized_bits[1]), persistent=False)

        all_codes = torch.arange(codebook_size)
        # indices_to_bits
        mask = 2 ** torch.arange(self.codebook_dim, dtype=torch.long)
        bits = (all_codes.unsqueeze(-1) & mask) != 0
        codebook = bits * 2.0 - 1.0
        self.register_buffer('codebook', codebook, persistent=False)

    def forward(self, x):
        x = rearrange(x, 'b d ... -> b ... d')
        x, ps = pack_one(x, 'b * d')
        x = rearrange(x, 'b n (c d) -> b n c d', c=self.num_codebooks)
        codebook_value = torch.Tensor([1.0]).to(device=x.device, dtype=x.dtype)
        quantized = torch.where(x > 0, codebook_value, -codebook_value)
        
        quantized = x + (quantized - x).detach()
        quantized = rearrange(quantized, 'b n c d -> b n (c d)')
        quantized = unpack_one(quantized, ps, 'b * d')
        quantized = rearrange(quantized, 'b ... d -> b d ...')
        
        return quantized, None, None, None

class VQModel(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = LFQ(dim=embed_dim, codebook_size=n_embed)

    def encode(self, x):
        h = self.encoder(x)
        quant, _, _, _ = self.quantize(h)
        return quant

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def calculate_psnr(img1, img2):
    # img1, img2: [H, W, 3] uint8 or float
    if img1.dtype != img2.dtype:
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def quant_to_indices(quant):
    # quant: [B, C, H, W] in {-1, 1}
    # returns: [B, H, W] int32 indices
    C = quant.shape[1]
    basis = 2 ** torch.arange(C, device=quant.device, dtype=torch.int32)
    basis = basis.view(1, C, 1, 1)
    bits = (quant > 0).int()
    indices = torch.sum(bits * basis, dim=1)
    return indices

def indices_to_quant(indices, C=18):
    # indices: [B, H, W] int32
    # returns: [B, C, H, W] float32 in {-1, 1}
    B, H, W = indices.shape
    device = indices.device
    basis = 2 ** torch.arange(C, device=device, dtype=torch.int32)
    indices_expanded = indices.unsqueeze(1)
    basis_expanded = basis.view(1, C, 1, 1)
    bits = (indices_expanded & basis_expanded) != 0
    quant = bits.float() * 2.0 - 1.0
    return quant

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def load_model(device):
    ddconfig = {
        "double_z": False,
        "z_channels": 18,
        "resolution": 128,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1,2,2,4],
        "num_res_blocks": 4,
    }
    n_embed = 262144
    embed_dim = 18
    repo_id = "TencentARC/Open-MAGVIT2-Tokenizer-128-resolution"
    filename = "imagenet_128_L.ckpt"
    
    print(f"Loading model checkpoint from {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

    model = VQModel(ddconfig, n_embed, embed_dim)
    
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        
        new_sd = {}
        for k, v in sd.items():
            if "loss" in k or "discriminator" in k or "lpips" in k:
                continue
            new_sd[k] = v
            
        model.load_state_dict(new_sd, strict=False)
        print("Model weights loaded.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    model.to(device)
    model.eval()
    return model

def process_image_to_tokens(model, img_path, output_path, device, args):
    print(f"Loading image: {img_path}")
    original_pil = Image.open(img_path).convert("RGB")
    original_w, original_h = original_pil.size
    
    # Candidates represent the "long edge" size
    candidates = [256, 384, 512, 640, 768, 1024, 1280, 1536, 1792, 2048]
    
    if args.size:
        candidates = [args.size]
    
    best_res = (0,0)
    best_psnr = -1.0
    best_quant = None
    best_rec_img = None
    
    search_logs = []

    print("Starting adaptive resolution search..." if not args.size else f"Using fixed resolution: {args.size}")
    
    for max_dim in candidates:
        # Calculate dimensions preserving aspect ratio
        scale = max_dim / max(original_w, original_h)
        t_W = int(original_w * scale)
        t_H = int(original_h * scale)
        
        # Ensure multiple of 8
        t_W = ((t_W + 7) // 8) * 8
        t_H = ((t_H + 7) // 8) * 8
        
        if t_W < 16 or t_H < 16:
            continue

        img_resized = original_pil.resize((t_W, t_H), Image.BICUBIC)
        
        # To Tensor
        x_np = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
        x = torch.from_numpy(x_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            quant = model.encode(x)
            rec = model.decode(quant)
            
        # Post-process reconstruction
        rec = rec.cpu().squeeze(0).permute(1, 2, 0).numpy()
        rec = (rec + 1.0) * 127.5
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        rec_pil = Image.fromarray(rec)
        
        # Resize back to ORIGINAL size for comparison
        rec_upscaled = rec_pil.resize((original_w, original_h), Image.BICUBIC)
        
        # Calculate PSNR against ORIGINAL image
        psnr = calculate_psnr(np.array(original_pil), np.array(rec_upscaled))
        
        print(f"  Max Edge {max_dim} -> Resolution {t_W}x{t_H} -> PSNR (vs Original): {psnr:.2f} dB")
        
        search_logs.append({
            "resolution": (t_W, t_H),
            "psnr": psnr,
            "tokens": quant.shape[2]*quant.shape[3],
            "rec_img": rec_pil if args.debug else None
        })
        
        best_res = (t_W, t_H)
        best_quant = quant
        best_psnr = psnr
        best_rec_img = rec_pil
        
        if psnr >= args.psnr:
            print(f"  [+] Target PSNR {args.psnr} dB reached.")
            break
            
    print(f"Final Selection: {best_res[0]}x{best_res[1]} (Tokens: {best_quant.shape[2]*best_quant.shape[3]}) with PSNR {best_psnr:.2f} dB")
    
    # Save Tokens
    indices = quant_to_indices(best_quant)
    indices_list = indices.flatten().cpu().tolist()
    
    data = {
        "original_size": (original_w, original_h),
        "resolution": best_res,
        "latent_shape": list(indices.shape),
        "tokens": indices_list
    }
    
    if output_path is None:
        base, _ = os.path.splitext(img_path)
        output_path = base + ".json"
        
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Saved tokens to {output_path}")
    print(f"Final Reconstruction PSNR: {best_psnr:.2f} dB")
    
    # Debug HTML
    if args.debug:
        demo_dir = "demo"
        if not os.path.exists(demo_dir):
            os.makedirs(demo_dir)
            
        input_name = "debug_input.png"
        rec_name = "debug_output.png"
        original_pil.save(os.path.join(demo_dir, input_name))
        best_rec_img.save(os.path.join(demo_dir, rec_name))
        
        html = f"""
        <html><body>
        <h1>Tokenization Debug Report</h1>
        <p><b>Original Image:</b> {img_path} ({original_w}x{original_h})</p>
        <p><b>Selected Resolution:</b> {best_res} (PSNR: {best_psnr:.2f} dB)</p>
        <p><b>Tokens:</b> {len(indices_list)}</p>
        <div style="display:flex; gap:20px;">
            <div><h3>Original</h3><img src="{input_name}" width="500"></div>
            <div><h3>Reconstruction</h3><img src="{rec_name}" width="500"></div>
        </div>
        <h3>Search Log</h3>
        <ul>
        """
        for log in search_logs:
            html += f"<li>{log['resolution']}: {log['psnr']:.2f} dB ({log['tokens']} tokens)</li>"
        html += "</ul></body></html>"
        
        with open(os.path.join(demo_dir, "index.html"), "w") as f:
            f.write(html)
        print(f"Debug report saved to {demo_dir}/index.html")

def process_tokens_to_image(model, json_path, output_path, device, args):
    print(f"Loading tokens from: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
        
    shape = data["latent_shape"]
    tokens = torch.tensor(data["tokens"], dtype=torch.int32).to(device)
    tokens = tokens.view(*shape)
    
    quant = indices_to_quant(tokens, C=18).to(device)
    
    print("Decoding tokens...")
    with torch.no_grad():
        rec = model.decode(quant)
        
    rec = rec.cpu().squeeze(0).permute(1, 2, 0).numpy()
    rec = (rec + 1.0) * 127.5
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    rec_img = Image.fromarray(rec)
    
    if "original_size" in data:
        orig_w, orig_h = data["original_size"]
        if (orig_w, orig_h) != rec_img.size:
             print(f"Restoring original size: {orig_w}x{orig_h}")
             rec_img = rec_img.resize((orig_w, orig_h), Image.BICUBIC)
    
    if output_path is None:
        base, _ = os.path.splitext(json_path)
        ext = ".avif" if args.avif else ".png"
        output_path = base + "_restored" + ext
        
    rec_img.save(output_path)
    print(f"Saved reconstructed image to {output_path}")

    # Try to find original image for PSNR calculation
    base_no_ext = os.path.splitext(json_path)[0]
    possible_extensions = [".png", ".jpg", ".jpeg", ".webp", ".avif"]
    ref_img = None
    for ext in possible_extensions:
        if os.path.exists(base_no_ext + ext):
            ref_img = Image.open(base_no_ext + ext).convert("RGB")
            break
    
    if ref_img is not None:
        if ref_img.size != rec_img.size:
            ref_img_resized = ref_img.resize(rec_img.size, Image.BICUBIC)
        else:
            ref_img_resized = ref_img
        psnr = calculate_psnr(np.array(ref_img_resized), np.array(rec_img))
        print(f"Reconstruction PSNR (vs original): {psnr:.2f} dB")

def main():
    parser = argparse.ArgumentParser(description="Open-MAGVIT2 Tokenizer CLI")
    parser.add_argument("input_path", help="Path to input image (for tokenization) or JSON file (for reconstruction)")
    parser.add_argument("-o", "--output", help="Path to output file")
    parser.add_argument("--size", type=int, help="Force specific resolution (skip auto-search)")
    parser.add_argument("--psnr", type=float, default=32.0, help="Target PSNR for auto-resolution (default: 32.0)")
    parser.add_argument("--debug", action="store_true", help="Generate HTML report (only for tokenization)")
    parser.add_argument("--avif", action="store_true", help="Use AVIF format for output images")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: File not found {args.input_path}")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = load_model(device)
    
    # Determine mode
    _, ext = os.path.splitext(args.input_path)
    if ext.lower() == ".json":
        process_tokens_to_image(model, args.input_path, args.output, device, args)
    else:
        process_image_to_tokens(model, args.input_path, args.output, device, args)

if __name__ == "__main__":
    main()