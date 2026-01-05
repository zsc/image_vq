import sys
import os
import math
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, pack, unpack
from PIL import Image
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# Utils & Modules from improved_model.py
# -----------------------------------------------------------------------------

def swish(x):
    return x*torch.sigmoid(x)

def depth_to_space(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """ Depth-to-Space DCR mode (depth-column-row) core implementation. """
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
        # calcuate var for scale
        scale = rearrange(quantizer, "b c h w -> b c (h w)")
        scale = scale.var(dim=-1) + self.eps 
        scale = scale.sqrt()
        scale = self.gamma(scale).view(B, C, 1, 1)

        # calculate mean for bias
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

# -----------------------------------------------------------------------------
# LFQ (from before)
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# VQModel
# -----------------------------------------------------------------------------

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
# Main
# -----------------------------------------------------------------------------

def calculate_psnr(img1, img2):
    # img1, img2: [H, W, 3] uint8
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def process_resolution(model, img, resolution, device, out_dir):
    # Ensure multiple of 8
    H, W = resolution, resolution # Square for now as per request
    H = ((H + 7) // 8) * 8
    W = ((W + 7) // 8) * 8
    
    img_resized = img.resize((W, H), Image.BICUBIC)
    
    x = np.array(img_resized).astype(np.float32) / 127.5 - 1.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0) # B C H W
    x = x.to(device)

    with torch.no_grad():
        quant = model.encode(x)
        rec = model.decode(quant)

    rec = rec.cpu().squeeze(0).permute(1, 2, 0).numpy()
    rec = (rec + 1.0) * 127.5
    rec = np.clip(rec, 0, 255).astype(np.uint8)
    rec_img = Image.fromarray(rec)
    
    # Calculate PSNR
    original_np = np.array(img_resized)
    psnr = calculate_psnr(original_np, rec)
    
    # Latent info
    latent_h, latent_w = quant.shape[2], quant.shape[3]
    num_tokens = latent_h * latent_w
    
    # Save images
    base_name = f"res_{W}x{H}"
    img_resized.save(os.path.join(out_dir, f"{base_name}_input.png"))
    rec_img.save(os.path.join(out_dir, f"{base_name}_output.png"))
    
    return {
        "resolution": (W, H),
        "latent_res": (latent_w, latent_h),
        "num_tokens": num_tokens,
        "psnr": psnr,
        "input_path": f"{base_name}_input.png",
        "output_path": f"{base_name}_output.png"
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Open-MAGVIT2 Inference Demo")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--size", type=int, help="Specific resolution to run (will be rounded up to multiple of 8)")
    args = parser.parse_args()

    print("Initializing...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

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
    # ckpt_path = "demo/Open-Magvit2-hf/imagenet_128_L.ckpt"
    repo_id = "TencentARC/Open-MAGVIT2-Tokenizer-128-resolution"
    filename = "imagenet_128_L.ckpt"
    print(f"Loading checkpoint {filename} from {repo_id}...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)

    model = VQModel(ddconfig, n_embed, embed_dim)
    
    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}")
        sd = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        
        new_sd = {}
        for k, v in sd.items():
            if "loss" in k or "discriminator" in k or "lpips" in k:
                continue
            new_sd[k] = v
            
        msg = model.load_state_dict(new_sd, strict=False)
        print(f"Weights loaded. {msg}")
    else:
        print(f"Checkpoint not found at {ckpt_path}. Please check path.")
        return

    model.to(device)
    model.eval()

    img_path = args.image_path
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return

    print(f"Processing {img_path}")
    img = Image.open(img_path).convert("RGB")
    out_dir = "demo"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    results = []
    
    resolutions_to_run = [64, 128, 192, 256, 512]
    if args.size:
        resolutions_to_run = [args.size]
    
    resolutions_to_run.sort()

    for res in resolutions_to_run:
        print(f"Running resolution: {res}x{res}")
        res_data = process_resolution(model, img, res, device, out_dir)
        results.append(res_data)
        print(f"  PSNR: {res_data['psnr']:.2f}")

    # Generate HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Open-MAGVIT2 Multi-Resolution Reconstruction</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 20px; }
            .container { display: flex; flex-direction: column; gap: 40px; align-items: center; }
            .row { border-bottom: 1px solid #eee; padding-bottom: 20px; }
            .images { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }
            .box { display: flex; flex-direction: column; align-items: center; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; }
            table { margin: 0 auto; border-collapse: collapse; margin-bottom: 30px; }
            th, td { border: 1px solid #ddd; padding: 8px 12px; }
            th { background-color: #f4f4f4; }
        </style>
    </head>
    <body>
        <h1>Open-MAGVIT2 Reconstruction Result</h1>
        <p>Model: imagenet_128_L | Downsampling Factor: 8</p>
        
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Resolution</th>
                <th>Latent Size</th>
                <th>Tokens</th>
                <th>PSNR</th>
            </tr>
    """
    
    for r in results:
        html_content += f"""
            <tr>
                <td>{r['resolution'][0]}x{r['resolution'][1]}</td>
                <td>{r['latent_res'][0]}x{r['latent_res'][1]}</td>
                <td>{r['num_tokens']}</td>
                <td>{r['psnr']:.2f} dB</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <div class="container">
    """
    
    for r in results:
        html_content += f"""
            <div class="row">
                <h3>Resolution: {r['resolution'][0]}x{r['resolution'][1]} (PSNR: {r['psnr']:.2f} dB)</h3>
                <p>Latent: {r['latent_res'][0]}x{r['latent_res'][1]} ({r['num_tokens']} tokens)</p>
                <div class="images">
                    <div class="box">
                        <h4>Input</h4>
                        <img src="{r['input_path']}" alt="Input {r['resolution'][0]}">
                    </div>
                    <div class="box">
                        <h4>Reconstructed</h4>
                        <img src="{r['output_path']}" alt="Reconstructed {r['resolution'][0]}">
                    </div>
                </div>
            </div>
        """
        
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html_content)
        
    print("Done. Saved results and index.html to demo/")

if __name__ == "__main__":
    main()
