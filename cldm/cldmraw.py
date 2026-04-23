import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from math import ceil
import copy
from cldm.dhi import FeatureExtractor


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        # self.input_hint_block = TimestepEmbedSequential(
        #     conv_nd(dims, hint_channels, 16, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 16, 16, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 16, 32, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     conv_nd(dims, 32, 32, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 32, 96, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     conv_nd(dims, 96, 96, 3, padding=1),
        #     nn.SiLU(),
        #     conv_nd(dims, 96, 256, 3, padding=1, stride=2),
        #     nn.SiLU(),
        #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        # )

        self.input_hint_block = TimestepEmbedSequential(
            FeatureExtractor(hint_channels),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def validation_step(self, batch, batch_idx):
        # 取得輸入
        x, cond = self.get_input(batch, self.first_stage_key)
        
        # 計算 Loss (調用 p_losses)
        # t 可以在驗證時固定或隨機，通常建議隨機以涵蓋所有階段
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        
        # 這裡會跑完你寫的所有 CVC, LPIPS 邏輯
        loss, loss_dict = self.forward(x, cond, t) 
        
        # 取得總 Loss (在你的 p_losses 最後一行 update 的)
        prefix = 'val'
        loss_val = loss_dict.get(f'{prefix}/loss_simple', loss)
        
        # 重要：Log 出去給 Optuna 抓取
        self.log('val/loss', loss_val, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        # 也可以順便 Log 其他指標觀察
        if f'{prefix}/loss_cvc_focus' in loss_dict:
            self.log('val/loss_cvc', loss_dict[f'{prefix}/loss_cvc_focus'])
            
        return loss_val

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs


class ControlLDM(LatentDiffusion):

    def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        self.cvc_weight = 5.2908
        self.attn_focus_weight = 9.0166
        # ... 原有代碼
        self.lpips_loss = LPIPS(net='vgg').eval() #初始化感知損失模型，VGG16 作為特徵提取的骨幹網路，評估模式不更新梯度
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        
        self.lpips_weight = 0.1975  # 你可以調整這個權重
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs) #調用了父類 LatentDiffusion 的 get_input，VAE 會將高維的像素圖像壓縮成低維的潛在表x_start
        control_mask = batch[self.control_key]
        if bs is not None:
            control_mask = control_mask[:bs]
        control_mask = control_mask.to(self.device)
        control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
        control_mask = control_mask.to(memory_format=torch.contiguous_format).float()

        control_image = (batch["jpg"] + 1.0) / 2.0
        if bs is not None:
            control_image = control_image[:bs]
        control_image = control_image.to(self.device)
        control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
        control_image = control_image.to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], c_concat_mask=[control_mask], c_concat_image=[control_image]) #Conditioning（C,條件引導信號）---cond字典內(ddpm.py)
    def on_train_batch_start(self, batch, batch_idx):
        # 只在整個訓練的最開始執行一次
        if self.global_step == 0 and batch_idx == 0:
            print("\n>>> [全面預檢] 正在模擬複雜 Loss 管線...")
            device = self.device
            try:
                # 1. 模擬數據 (Latent 空間: 64x64, 4 channels)
                dummy_pred = torch.randn(2, 4, 64, 64, device=device).requires_grad_(True)
                dummy_gt = torch.randn(2, 4, 64, 64, device=device)
                dummy_mask = torch.randint(0, 2, (2, 1, 512, 512), device=device).float()

                # 2. 測試 Attention Map 邏輯 (包含 Gaussian Blur)
                mask_interp = torch.nn.functional.interpolate(dummy_mask, size=(64, 64), mode="nearest")
                from torchvision.transforms import functional as TF
                # 測試 Gaussian Blur 是否會因為半精度(fp16)報錯
                test_blur = TF.gaussian_blur(mask_interp, kernel_size=[5, 5], sigma=[1.5, 1.5])
                print(">>> [Success] Gaussian Blur 測試通過")

                # 3. 測試 clDice
                # 注意：clDice 通常預期輸入在 [0, 1]，這裡先做 sigmoid
                cldice_val = self.get_clDice_loss(torch.sigmoid(dummy_pred[:, :1]), mask_interp)
                print(f">>> [Success] clDice 測試通過: {cldice_val.item():.4f}")

                # 4. 測試 ROI 裁切與 LPIPS (最容易報錯的地方)
                # 模擬 decode 後的影像 (VAE 輸出通常是 512x512)
                img_decoded = torch.randn(2, 3, 512, 512, device=device)
                gt_decoded = torch.randn(2, 3, 512, 512, device=device)
                
                # 測試裁切函數
                crop_pred = self.crop_to_mask(img_decoded, dummy_mask)
                crop_gt = self.crop_to_mask(gt_decoded, dummy_mask)
                
                if crop_pred.shape != (2, 3, 224, 224):
                    raise ValueError(f"裁切尺寸不正確: {crop_pred.shape}")
                
                # 測試 LPIPS
                lpips_score = self.lpips_loss(crop_pred, crop_gt).mean()
                print(f">>> [Success] ROI + LPIPS 測試通過: {lpips_score.item():.4f}")

                print(">>> [All Clear] 所有損失函數邏輯檢查完畢，正式開始訓練。\n")

            except Exception as e:
                print(f"\n❌ [致命錯誤] 預檢失敗！錯誤訊息: {e}")
                import traceback
                traceback.print_exc()
                raise e # 終止訓練，避免浪費時間
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            if 'c_concat_image' in cond: #讓模型學會如何處理 Image 特徵，
                control_model_mask = copy.deepcopy(self.control_model).requires_grad_(False)
                diffusion_model_image = copy.deepcopy(diffusion_model)         
                control_weights_mask = 1.0
                control_weights_image = 1.0 * self.global_step / self.trainer.max_steps

                guidance_strength = 1.5 #引導強度參數，增強 Mask（遮罩）對生成結果的控制力

                control_image = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat_image'], 1), timesteps=t, context=cond_txt)
                control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]
                with torch.no_grad():
                    control_mask = control_model_mask(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
                    control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]
                
                #control = [control_weights_mask * c_mask.detach() + control_weights_image * c_image for c_mask, c_image in zip(control_mask, control_image)]
                control = [
                    (control_weights_image * c_image) + 
                    (control_weights_mask * guidance_strength * c_mask.detach()) 
                    for c_mask, c_image in zip(control_mask, control_image)
                ] #Mask 和 Image 基礎權重不同

                eps = diffusion_model_image(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
            else:
                control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat_mask, c_cat_image, c = c["c_concat_mask"][0][:N], c["c_concat_image"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["control_mask"] = c_cat_mask * 2.0 - 1.0
        log["control_image"] = c_cat_image * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((384, 384), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat_mask  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"] = x_samples_cfg

            # uc_cat_image = c_cat_image  # torch.zeros_like(c_cat_1)
            # uc_full = {"c_concat": [uc_cat], "c_concat_image": [uc_cat_image], "c_crossattn": [uc_cross]}
            # samples_cfg_image, _ = self.sample_log(cond={"c_concat": [c_cat_mask], "c_concat_image": [c_cat_image], "c_crossattn": [c]},
            #                                  batch_size=N, ddim=use_ddim,
            #                                  ddim_steps=ddim_steps, eta=ddim_eta,
            #                                  unconditional_guidance_scale=unconditional_guidance_scale,
            #                                  unconditional_conditioning=uc_full,
            #                                  )
            # x_samples_cfg_image = self.decode_first_stage(samples_cfg_image)
            # log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_image"] = x_samples_cfg_image

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()


    def soft_erode(self, img):
        # 針對 2D 影像的腐蝕操作
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        # 針對 2D 影像的膨脹操作
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        # 開運算：先腐蝕後膨脹，用來去除噪點
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img, iter_):
        # 真正的軟骨架提取：利用原圖減去開運算的結果，抓出細長特徵
        img1 = self.soft_open(img) #1. 抓出最外層的細微特徵 (第一次禮帽運算
        skel = F.gelu(img - img1) #GELU(可微分)： 它像一個平滑的門口。當差值為負時，它會將其壓向 0（過濾雜訊）；當差值為正時，它允許這個「骨架訊號」通過。確保了 skel 張量中只包含有效的結構資訊。
        for _ in range(iter_):
            img = self.soft_erode(img) # 2. 把影像「剝瘦一層」 (腐蝕)
            img1 = self.soft_open(img) #3. 在變瘦的影像上，再做一次禮帽運算，抓出這一層的「芯」
            delta = F.gelu(img - img1)
            skel = skel + F.gelu(delta - skel * delta)# 4. 把每一層抓到的「芯」全部疊加起來 (聯集)只填補尚未覆蓋的區域
        return skel
    def soft_dice(self, y_true, y_pred, smooth=1e-6):
        intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
        coeff = (2. * intersection + smooth) / (
            torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) + smooth
        )
        return 1. - coeff.mean()

    def get_clDice_loss(self, pred_mask, gt_mask, alpha=0.5, iter_=3):
        """
        融合 Dice 與 clDice 的損失函數
        alpha: 調整 Dice 與 clDice 的權重分配
        """
        # 1. 計算基礎 Dice Loss (確保管路粗細大致正確)
        dice_loss = self.soft_dice(gt_mask, pred_mask)

        # 2. 提取兩者的軟骨架
        s_p = self.soft_skel(pred_mask, iter_)
        s_l = self.soft_skel(gt_mask, iter_)

        # 3. 計算拓撲精確度 (Topology Precision/Sensitivity)
        # s_p * gt_mask: 預測的骨架有多少落在真實的導管區域內
        tprec = (torch.sum(s_p * gt_mask) + 1e-6) / (torch.sum(s_p) + 1e-6)
        # s_l * pred_mask: 真實的骨架有多少落在預測的導管區域內
        tsens = (torch.sum(s_l * pred_mask) + 1e-6) / (torch.sum(s_l) + 1e-6)
        
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens + 1e-6)

        # 4. 依照 alpha 混合
        total_loss = (1.0 - alpha) * dice_loss + alpha * cl_dice
        
        return total_loss
    def crop_to_mask(self, img, mask, pad=10):
        B, C, H, W = img.shape
        crops = []
        target_size = (224, 224) # 統一目標尺寸
    
        for b in range(B):
            m = mask[b, 0] > 0.5
            coords = m.nonzero(as_tuple=False)
        
        # 1. 如果沒抓到座標，直接對整張原圖縮放
            if coords.shape[0] == 0:
            # 修改點：不能直接 append 原圖，要縮放到 target_size
                full_img_resized = torch.nn.functional.interpolate(
                img[b:b+1], size=target_size, mode="bilinear", align_corners=False
                )
                crops.append(full_img_resized)
                continue
        
        # 2. 正常裁切邏輯
            y_min, x_min = coords.min(dim=0)[0]
            y_max, x_max = coords.max(dim=0)[0]
    
        # 加上 Padding 並防止越界
        # 注意：y_max/x_max 在切片時是不包含後邊界的，所以加 pad 後要確保範圍正確
            y_min_pad = max(y_min - pad, 0)
            y_max_pad = min(y_max + pad, H)
            x_min_pad = max(x_min - pad, 0)
            x_max_pad = min(x_max + pad, W)
        
        # 如果加了 pad 後區域太小（例如標註只有一個點），也要確保能裁切
            cropped = img[b:b+1, :, y_min_pad:y_max_pad, x_min_pad:x_max_pad]
        
        # 3. 縮放到 target_size
            cropped_resized = torch.nn.functional.interpolate(
            cropped, size=target_size, mode="bilinear", align_corners=False
        )
            crops.append(cropped_resized)

    # 現在 crops 裡的所有 Tensor 都是 (1, C, 224, 224)，可以安心 cat 了
        return torch.cat(crops, dim=0)

    def p_losses(self, x_start, cond, t, noise=None):

        cond_mask = {}
        cond_mask["c_crossattn"] = [cond["c_crossattn"][0]]
        cond_mask["c_concat"] = [cond["c_concat_mask"][0]]

        cond_image = {}
        cond_image["c_crossattn"] = [cond["c_crossattn"][0]]
        cond_image["c_concat"] = [cond["c_concat_mask"][0]]
        cond_image["c_concat_image"] = [cond["c_concat_image"][0]]

        weights_ones = torch.ones_like(t, device=x_start.device, dtype=x_start.dtype) #建立一個形狀（Shape）跟變數 t(Timestep）一樣的張量，且裡面的值全部填滿 1.0
        weights_thre = (t <= 200).to(device=x_start.device, dtype=x_start.dtype) #控制正則化損失(線上擴增)的開關:建立時間步布林遮罩，將剛剛產生的布林張量移至與輸入影像 x_start 相同的運算裝置上(卻抱張量運算在同裝置)

        weights_mask = 1.0 * weights_ones  # Loss 0
        weights_image = 1.0 * weights_ones  # Loss 1
        weights_mask_2_image = 1.0 * weights_ones  # Loss 2
        weights_mask_regularization = 1.0 * weights_thre  # Loss 3

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output_mask = self.apply_model(x_noisy, t, cond_mask) #模型預測出的噪聲

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        loss_dict.update({
        f'{prefix}/loss_lpips': torch.tensor(0.0, device=x_start.device),
        f'{prefix}/loss_cldice': torch.tensor(0.0, device=x_start.device),
        f'{prefix}/loss_roi_zoom_mse': torch.tensor(0.0, device=x_start.device),
        f'{prefix}/loss_roi_cldice': torch.tensor(0.0, device=x_start.device),
        })
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise ##
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = weights_mask * self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
        print(f"loss_simple_mask: {loss_simple.mean()}")
        #
        mask_for_attn = cond["c_concat_mask"][0].to(device=x_start.device, dtype=x_start.dtype) #字典中的鍵中的第一個值(硬體、精度對齊)
        if mask_for_attn.max() > 1.0:
            mask_for_attn = mask_for_attn / 255.0
            mask_for_attn = (mask_for_attn > 0.5).float()
        # 縮放遮罩尺寸以符合影像
        if mask_for_attn.shape[-2:] != x_start.shape[-2:]: #如果遮罩與目前正在計算 Loss 的影像高度 和寬度不同
            mask_for_attn = torch.nn.functional.interpolate( #最近鄰插值（縮放）遮罩的影像高度 和寬度
                mask_for_attn,
                size=x_start.shape[-2:],
                mode="nearest"
            )
        # 轉為單通道
        if mask_for_attn.shape[1] != 1:
            mask_for_attn = mask_for_attn.mean(dim=1, keepdim=True) #在第 1 個維度（也就是 Channel 維度）進行壓縮，轉單通道

        # 定義模糊函數(空間注意力圖)
        def gaussian_blur(x, sigma=2.0):
            kernel_size = int(2 * ceil(2 * sigma) + 1)
            # 建立高斯核 (這裡簡化使用內建卷積，確保在 GPU 上執行)
            # 你也可以直接用 torchvision.transforms.functional.gaussian_blur
            from torchvision.transforms import functional as TF
            return TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

        # 對原始 binary mask 先做模糊，再計算權重 map
        smoothed_mask = gaussian_blur(mask_for_attn, sigma=1.5) 
        
        # 建立權重矩陣：現在權重會從 (1 + focus_weight) 緩慢降到 1.0
        attn_map = smoothed_mask * self.attn_focus_weight + 1.0
        pixel_error = (model_output_mask - target)**2 #每個像素的平方誤差
        loss_cvc_focus = (
                        (pixel_error * attn_map).sum(dim=[1, 2, 3]) /
                        (attn_map.sum(dim=[1, 2, 3]) * pixel_error.shape[1]).clamp_min(1e-6)
                        )#如果該像素在遮罩上，它的誤差會乘以 6。如果該像素在背景，它的誤差維持不變（乘以 1）。
        loss_cvc_focus = loss_cvc_focus.to(x_start.dtype)
        prefix = 'train' if self.training else 'val'
        loss_dict.update({f'{prefix}/loss_cvc_focus': loss_cvc_focus.mean()})
        
        # 將此損失加入總和 (給予 0.5 的權重，可視實驗調整)
        loss_simple = loss_simple + self.cvc_weight * loss_cvc_focus
        print(f">>> Current CVC Weight: {self.cvc_weight}")
        print(f"loss_cvc_focus: {loss_cvc_focus.mean():.6f}")

        # 新增：Masked LPIPS 感知損失
        # ==========================================
        # 只有在適當的 timestep 或是訓練後期開啟，以節省顯存並確保結構已定型
        loss_lpips = torch.tensor(0.0, device=x_start.device)
        loss_cldice = torch.tensor(0.0, device=x_start.device)
        is_late_training = (self.global_step > (self.trainer.max_steps * 1 / 3)) or (self.trainer.max_steps < 2000)    
        is_low_noise_step = (t.float().mean() < 600) # LPIPS 在低雜訊時計算才有意義
        
        if hasattr(self, "lpips_loss") and is_late_training and is_low_noise_step:
            # 1. 預測 z0
            pred_z0 = self.predict_start_from_noise(x_noisy, t, model_output_mask) if self.parameterization == "eps" else model_output_mask
            
            # 2. 解碼 (確保在 GPU 且不計算梯度以節省顯存，除非你要 backprop lpips)
            out_pixel = self.decode_first_stage(pred_z0) #$$[Batch, Channel, Height, Width]$$
            target_pixel = self.decode_first_stage(x_start)
            
            # --- [新增：clDice 計算部分] ---
            # 將 RGB/多通道轉為單通道機率圖 (0~1)
            # 假設 CVC 在解碼後的影像中是白色的線條
            v_p = torch.mean(out_pixel, dim=1, keepdim=True)
            v_p = (v_p + 1.0) / 2.0 

# 重要修正：不要直接拿縮小過的 mask_for_attn
# 應該拿原始 batch 裡的 mask，或者將 mask_for_attn 放大回 out_pixel 的尺寸
            v_l = cond["c_concat_mask"][0].to(device=x_start.device, dtype=x_start.dtype)

            if v_l.max() > 1.0:
                v_l = v_l / 255.0

# 確保 v_l (Label) 的尺寸跟 v_p (Pred) 一模一樣
            if v_l.shape[-2:] != v_p.shape[-2:]:
                v_l = torch.nn.functional.interpolate(
                v_l, 
                size=v_p.shape[-2:], 
                mode="nearest" # Mask 建議用 nearest 保持邊緣銳利
                )

# 為了 clDice 效果，稍微強化一下預測圖的對比
            v_p = torch.sigmoid((v_p - 0.5) * 15.0)

# 現在兩者都是 (B, 1, 384, 384)，計算就不會報錯了
            loss_cldice = self.get_clDice_loss(v_p, v_l, alpha=0.7, iter_=5)

            # 3. 處理 Mask
            lpips_mask = torch.nn.functional.interpolate(
                mask_for_attn, size=out_pixel.shape[-2:], mode="bilinear"
            ).detach()

            # B. 聚焦效果：將影像與 Mask 相乘
            # 這樣 LPIPS 就只會比對導管區域的質地，忽略背景 X-ray 雜訊
            cropped_out = self.crop_to_mask(out_pixel, lpips_mask)
            cropped_target = self.crop_to_mask(target_pixel, lpips_mask)

            loss_lpips = self.lpips_loss(cropped_out, cropped_target).mean()
            # 4. 計算 Masked LPIPS
            # 先遮罩再算感知差異，能讓模型更專注於細菌主體
            lpips_lambda = getattr(self, "lpips_weight", 0.1)
            cldice_lambda = 1.0  # 建議初始值 0.3，確保拓撲結構
    
            
    
            loss_simple = loss_simple + lpips_lambda * loss_lpips + cldice_lambda * loss_cldice

            # 更新 Dict 供 Log 使用
            loss_dict.update({f'{prefix}/loss_lpips': loss_lpips})
            loss_dict.update({f'{prefix}/loss_cldice': loss_cldice})
            print(f">>> loss_lpips: {loss_lpips:.6f} | loss_cldice: {loss_cldice:.6f}")

        # === 解決斷裂：全域與局部的拉扯 (Topology-Preserved ROI) ===
        if is_late_training and is_low_noise_step:
            with torch.no_grad():
                binary_mask = (mask_for_attn > 0.5).float() #將遮罩轉為純 0 或 1 $$[B, C, H, W]$$
                
                if binary_mask.sum() > 10: #如果一張圖裡的導管標註不到 10 個像素（太小或根本沒標註），就跳過 ROI 計算
                    # 1. 取得 BBox 座標
                    mask_any_h = binary_mask.any(dim=-1).any(dim=1) # 橫向投影：檢查每一列是否有導管(兩次「降維壓縮[B, C, H]->[B,H])
                    mask_any_w = binary_mask.any(dim=-2).any(dim=1) # 縱向投影：檢查每一欄是否有導管
                    y_indices = torch.where(mask_any_h)[1] # 找出所有 True 的高度索引
                    x_indices = torch.where(mask_any_w)[1]
                    
                    y_min, y_max = y_indices.min(), y_indices.max() # 最小值就是頂部，最大值就是底部
                    x_min, x_max = x_indices.min(), x_indices.max()

                    # 稍微擴大邊界確保導管邊界連貫，避免導管的末端或邊緣可能會剛好被切在邊界上
                    pad = 25 
                    h_img, w_img = out_pixel.shape[-2:] #倒數第二個維度」開始，一直取到最後
                    y_min, y_max = max(0, y_min - pad), min(h_img, y_max + pad)
                    x_min, x_max = max(0, x_min - pad), min(w_img, x_max + pad)

            # 2. 執行局部放大
            roi_out = out_pixel[:, :, y_min:y_max, x_min:x_max] #[Batch, Channel, Height, Width]
            roi_target = target_pixel[:, :, y_min:y_max, x_min:x_max]
            #裁出來的矩形正規化
            roi_out_zoom = torch.nn.functional.interpolate(roi_out, size=(224, 224), mode="bilinear")
            roi_target_zoom = torch.nn.functional.interpolate(roi_target, size=(224, 224), mode="bilinear")

            # 3. [關鍵] 計算 ROI 區域的 clDice (防止局部斷裂)
            # 將 ROI 影像轉為單通道機率圖
            v_p_roi = torch.mean(roi_out_zoom, dim=1, keepdim=True)
            v_p_roi = torch.sigmoid(( (v_p_roi + 1.0)/2.0 - 0.5) * 10.0) # 強對比處理(乘上 10.0 的作用： Sigmoid 在 $0$ 附近的變化最劇烈。乘上 $10$ 會讓這條 S 曲線變得非常陡峭。)
            # 從乾淨的 binary mask 中裁切出 ROI 作為目標
            roi_mask = mask_for_attn[:, :, y_min:y_max, x_min:x_max]
            v_l_roi = torch.nn.functional.interpolate(roi_mask, size=(224, 224), mode="nearest").detach()

            # alpha=0.9 代表極度重視「連通性」而非「像素值」
            loss_cldice_roi = self.get_clDice_loss(v_p_roi, v_l_roi, alpha=0.9, iter_=3)

            # 4. 計算 ROI MSE (細節質地)
            loss_roi_mse = torch.nn.functional.mse_loss(roi_out_zoom, roi_target_zoom)
            
            # 5. 組合損失：給予 ROI clDice 較高的權重來防止斷裂
            roi_lambda = 0.4        # 局部細節權重
            roi_topo_lambda = 1.0   # 局部連貫權重 (核心守護者)
            
            loss_simple = loss_simple + (roi_lambda * loss_roi_mse) + (roi_topo_lambda * loss_cldice_roi)
            
            # Log 記錄
            loss_dict.update({f'{prefix}/loss_roi_zoom_mse': loss_roi_mse})
            loss_dict.update({f'{prefix}/loss_roi_cldice': loss_cldice_roi})
            print(f">>> ROI Zoom - MSE: {loss_roi_mse:.6f} | ROI clDice: {loss_cldice_roi:.6f}")

        if weights_image.all():
            model_output_image = self.apply_model(x_noisy, t, cond_image)
            loss_simple_image = self.get_loss(model_output_image, target, mean=False).mean([1, 2, 3])
            print(f"loss_simple_image: {loss_simple_image.mean()}")
            loss_simple = loss_simple + weights_image * loss_simple_image

        if weights_mask_2_image.all():
            loss_simple_mask_2_image = self.get_loss(model_output_mask, model_output_image.detach(), mean=False).mean([1, 2, 3])
            print(f"loss_simple_mask_2_image: {loss_simple_mask_2_image.mean()}")
            loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image

        # if (self.global_step > (self.trainer.max_steps * 1 / 3)) and weights_mask_regularization.any(): # Done!
        #      noise_image_2_mask = torch.randn_like(x_start) #在訓練的每一個 Step，即時（On-the-fly）產生模型沒看過的資料變體。
        #      #recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
        #      x_noisy_mask_recon = self.q_sample(x_start=x_start,  t=t, noise=noise_image_2_mask) # 使用真實影像 (x_start) 加上一組全新的隨機雜訊，重新產生一個含噪樣本 (x_t)

        #      model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask) #凍結加噪的過程
        #      loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
        #      print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean()}")
        #      loss_simple = loss_simple + 0.5 * weights_mask_regularization * loss_simple_mask_regularization
        
        #  current_progress = self.global_step / self.trainer.max_steps
        #  if current_progress > (1/3) and weights_mask_regularization.any():
            
        #      # 判斷目前是在階段 2 還是階段 3
        #      if current_progress <= (2/3):
        #          # 【階段 2】中期：使用真實影像 x_start 確保結構對齊 (穩定地基)
        #         target_recon = x_start
        #         print(f"--- Phase 2: Using x_start for Regularization (Step {self.global_step}) ---")
        #      else:
        #          # 【階段 3】後期：使用 Image 分支預測的偽影像進行知識蒸餾 (強化細節)
        #          model_output_image = self.apply_model(x_noisy, t, cond_image)
        #          target_recon = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image) 
        #          print(f"--- Phase 3: Using recon_output_image for Distillation (Step {self.global_step}) ---")

        #      # 重新加噪並預測
        #      noise_image_2_mask = torch.randn_like(x_start).to(x_start.device)
        #      # 這裡的 x_start 根據階段切換為 target_recon
        #      x_noisy_mask_recon = self.q_sample(x_start=target_recon.detach(), t=t, noise=noise_image_2_mask) #凍結「加噪後的樣本」
            
        #      model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask) 
        #      loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
        #      print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean()}")
        #      # 加入總損失
        #      loss_simple = loss_simple + 0.5 * weights_mask_regularization * loss_simple_mask_regularization
        
        if (self.global_step > (self.trainer.max_steps * 1 / 3)) and weights_mask_regularization.any(): # Done!
             recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
             
             noise_image_2_mask = default(noise, lambda: torch.randn_like(recon_output_image))
             x_noisy_mask_recon = self.q_sample(x_start=recon_output_image, t=t, noise=noise_image_2_mask)
             recon_output_image = torch.clamp(recon_output_image, -1.0, 1.0)
             model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask)
             loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
             print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean()}")
             loss_simple = loss_simple +0.1* weights_mask_regularization * loss_simple_mask_regularization

 # --- 之前的代碼 (loss_simple 已經累加了 CVC, LPIPS, Regularization) ---
        
        # 1. 建立 loss_dict 基礎 (這對 Log 到 TensorBoard/Optuna 很重要)
        loss_simple_vlb = self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])

        # 2. 建立 loss_dict 基礎 (記錄未加權的主損失)
        loss_dict.update({f'{prefix}/loss_simple_raw': loss_simple.mean()})

        # 3. 計算 Adaptive Variance (logvar) 權重下的損失
        # 我們維持使用強化過的 loss_simple 來引導梯度，但 VLB 部分獨立出來
        logvar_t = self.logvar[t].to(self.device)
        
        # 這裡的主損失 loss 包含所有細菌強化的項，並透過 logvar 自動縮放權重
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        # 4. 最終縮放
        loss = self.l_simple_weight * loss.mean()

        # 5. 【關鍵修改】VLB 計算使用「乾淨」的損失
        # 使用 loss_simple_vlb (純 MSE) 而非包含 x6 權重的 loss_simple
        loss_vlb = (self.lvlb_weights[t] * loss_simple_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        
        # 累加 VLB 到最終 Loss
        loss += (self.original_elbo_weight * loss_vlb)
        
        # 6. 更新最終標籤供 Log 使用
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict