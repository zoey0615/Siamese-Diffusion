# #沒底圖有損失
# import einops
# import torch
# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F
# from math import ceil
# from ldm.modules.diffusionmodules.util import (
#     conv_nd,
#     linear,
#     zero_module,
#     timestep_embedding,
# )

# from einops import rearrange, repeat
# from torchvision.utils import make_grid
# from ldm.modules.attention import SpatialTransformer
# from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# from ldm.models.diffusion.ddpm import LatentDiffusion
# from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
# from ldm.models.diffusion.ddim import DDIMSampler

# import copy
# from cldm.dhi import FeatureExtractor


# class ControlledUnetModel(UNetModel):
#     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
#         hs = []
#         with torch.no_grad():
#             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#             emb = self.time_embed(t_emb)
#             h = x.type(self.dtype)
#             for module in self.input_blocks:
#                 h = module(h, emb, context)
#                 hs.append(h)
#             h = self.middle_block(h, emb, context)

#         if control is not None:
#             h += control.pop()

#         for i, module in enumerate(self.output_blocks):
#             if only_mid_control or control is None:
#                 h = torch.cat([h, hs.pop()], dim=1)
#             else:
#                 h = torch.cat([h, hs.pop() + control.pop()], dim=1)
#             h = module(h, emb, context)

#         h = h.type(x.dtype)
#         return self.out(h)


# class ControlNet(nn.Module):
#     def __init__(
#             self,
#             image_size,
#             in_channels,
#             model_channels,
#             hint_channels,
#             num_res_blocks,
#             attention_resolutions,
#             dropout=0,
#             channel_mult=(1, 2, 4, 8),
#             conv_resample=True,
#             dims=2,
#             use_checkpoint=False,
#             use_fp16=False,
#             num_heads=-1,
#             num_head_channels=-1,
#             num_heads_upsample=-1,
#             use_scale_shift_norm=False,
#             resblock_updown=False,
#             use_new_attention_order=False,
#             use_spatial_transformer=False,  # custom transformer support
#             transformer_depth=1,  # custom transformer support
#             context_dim=None,  # custom transformer support
#             n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
#             legacy=True,
#             disable_self_attentions=None,
#             num_attention_blocks=None,
#             disable_middle_self_attn=False,
#             use_linear_in_transformer=False,
#     ):
#         super().__init__()
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         if num_heads == -1:
#             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

#         if num_head_channels == -1:
#             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

#         self.dims = dims
#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         if isinstance(num_res_blocks, int):
#             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
#         else:
#             if len(num_res_blocks) != len(channel_mult):
#                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
#                                  "as a list/tuple (per-level) with the same length as channel_mult")
#             self.num_res_blocks = num_res_blocks
#         if disable_self_attentions is not None:
#             # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
#             assert len(disable_self_attentions) == len(channel_mult)
#         if num_attention_blocks is not None:
#             assert len(num_attention_blocks) == len(self.num_res_blocks)
#             assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
#             print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
#                   f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
#                   f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
#                   f"attention will still not be set.")

#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#         self.predict_codebook_ids = n_embed is not None

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
#                 )
#             ]
#         )
#         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

#         # self.input_hint_block = TimestepEmbedSequential(
#         #     conv_nd(dims, hint_channels, 16, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 16, 16, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 16, 32, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 32, 32, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 32, 96, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 96, 96, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 96, 256, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         # )

#         self.input_hint_block = TimestepEmbedSequential(
#             FeatureExtractor(hint_channels),
#             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         )

#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for nr in range(self.num_res_blocks[level]):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         # num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     if exists(disable_self_attentions):
#                         disabled_sa = disable_self_attentions[level]
#                     else:
#                         disabled_sa = False

#                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
#                         layers.append(
#                             AttentionBlock(
#                                 ch,
#                                 use_checkpoint=use_checkpoint,
#                                 num_heads=num_heads,
#                                 num_head_channels=dim_head,
#                                 use_new_attention_order=use_new_attention_order,
#                             ) if not use_spatial_transformer else SpatialTransformer(
#                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
#                                 use_checkpoint=use_checkpoint
#                             )
#                         )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             # num_heads = 1
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
#                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
#                 use_checkpoint=use_checkpoint
#             ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self.middle_block_out = self.make_zero_conv(ch)
#         self._feature_size += ch

#     def make_zero_conv(self, channels):
#         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

#     def forward(self, x, hint, timesteps, context, **kwargs):
#         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#         emb = self.time_embed(t_emb)

#         guided_hint = self.input_hint_block(hint, emb, context)

#         outs = []

#         h = x.type(self.dtype)
#         for module, zero_conv in zip(self.input_blocks, self.zero_convs):
#             if guided_hint is not None:
#                 h = module(h, emb, context)
#                 h += guided_hint
#                 guided_hint = None
#             else:
#                 h = module(h, emb, context)
#             outs.append(zero_conv(h, emb, context))

#         h = self.middle_block(h, emb, context)
#         outs.append(self.middle_block_out(h, emb, context))

#         return outs


# class ControlLDM(LatentDiffusion):

#     def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.control_model = instantiate_from_config(control_stage_config)
#         self.control_key = control_key
#         self.only_mid_control = only_mid_control
#         self.control_scales = [1.0] * 13 #ControlNet 提取出的「特徵組數」，用來對應並注入到 UNet 的特定層級中。
#         self.cvc_weight = 0.05
#         self.attn_focus_weight = 0.02
#     @torch.no_grad()
#     def get_input(self, batch, k, bs=None, *args, **kwargs):
#         x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
#         control_mask = batch[self.control_key]
#         if bs is not None:
#             control_mask = control_mask[:bs]
#         control_mask = control_mask.to(self.device)
#         control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
#         control_mask = control_mask.to(memory_format=torch.contiguous_format).float()

#         control_image = (batch["jpg"] + 1.0) / 2.0
#         if bs is not None:
#             control_image = control_image[:bs]
#         control_image = control_image.to(self.device)
#         control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
#         control_image = control_image.to(memory_format=torch.contiguous_format).float()

#         return x, dict(c_crossattn=[c], c_concat_mask=[control_mask], c_concat_image=[control_image])

#     def apply_model(self, x_noisy, t, cond, *args, **kwargs):
#         assert isinstance(cond, dict)
#         diffusion_model = self.model.diffusion_model

#         cond_txt = torch.cat(cond['c_crossattn'], 1) #文字條件

#         if cond['c_concat'] is None: #Conditioning by Concatenation
#             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
#         else:
#             if 'c_concat_image' in cond: #如果在推論階段給影像提示
#                 control_model_mask = copy.deepcopy(self.control_model).requires_grad_(False)
#                 diffusion_model_image = copy.deepcopy(diffusion_model)         
#                 control_weights_mask = 1.0
#                 # --- 這裡改用 try-except 來徹底避開 Lightning 的屬性存取報錯 ---
#                 try:
#                     # 只有在正式 Trainer 環境下，這兩行才能執行成功
#                     t_max = self.trainer.max_steps
#                     g_step = self.global_step
#                     control_weights_image = 1.0 * g_step / t_max
#                 except (AttributeError, RuntimeError, Exception):
#                     # 如果在推理腳本中，存取上述變數會拋出異常，則直接給預設值
#                     control_weights_image = 1.0
#                 control_image = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat_image'], 1), timesteps=t, context=cond_txt)
#                 control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]
#                 with torch.no_grad():
#                     control_mask = control_model_mask(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
#                     control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]
#                 control = [control_weights_mask * c_mask.detach() + control_weights_image * c_image for c_mask, c_image in zip(control_mask, control_image)]
#                 eps = diffusion_model_image(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
#             else:
#                 control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
#                 control = [c * scale for c, scale in zip(control, self.control_scales)]
#                 eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

#         return eps

#     @torch.no_grad()
#     def get_unconditional_conditioning(self, N):
#         return self.get_learned_conditioning([""] * N)

#     @torch.no_grad()
#     def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
#                    quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
#                    plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
#                    use_ema_scope=True,
#                    **kwargs):
#         use_ddim = ddim_steps is not None

#         log = dict()
#         z, c = self.get_input(batch, self.first_stage_key, bs=N)
#         c_cat_mask, c_cat_image, c = c["c_concat_mask"][0][:N], c["c_concat_image"][0][:N], c["c_crossattn"][0][:N]
#         N = min(z.shape[0], N)
#         n_row = min(z.shape[0], n_row)
#         log["control_mask"] = c_cat_mask * 2.0 - 1.0
#         log["control_image"] = c_cat_image * 2.0 - 1.0
#         log["conditioning"] = log_txt_as_img((384, 384), batch[self.cond_stage_key], size=16)
#         if "filename" in batch:
#             raw = batch["filename"][:N]
#             # rearrange 維度以符合 PyTorch 格式
#             raw = rearrange(raw, 'b h w c -> b c h w').to(self.device)
#             log["0_Original_RAW"] = raw # 命名為 0_ 開頭會排在資料夾最前面
#         if plot_diffusion_rows:
#             # get diffusion row
#             diffusion_row = list()
#             z_start = z[:n_row]
#             for t in range(self.num_timesteps):
#                 if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
#                     t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
#                     t = t.to(self.device).long()
#                     noise = torch.randn_like(z_start)
#                     z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
#                     diffusion_row.append(self.decode_first_stage(z_noisy))

#             diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
#             diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
#             diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
#             diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
#             log["diffusion_row"] = diffusion_grid

#         if sample:
#             samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c],"c_concat_image": [c_cat_image]}, # <--- 必須加上這一行！
#                                                      batch_size=N, ddim=use_ddim,
#                                                      ddim_steps=ddim_steps, eta=ddim_eta)
#             x_samples = self.decode_first_stage(samples)
#             log["samples"] = x_samples
#             if plot_denoise_rows:
#                 denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
#                 log["denoise_row"] = denoise_grid

#         if unconditional_guidance_scale > 1.0:
#             uc_cross = self.get_unconditional_conditioning(N)
#             uc_cat = c_cat_mask  # torch.zeros_like(c_cat)
#             uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
#             samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c]},
#                                              batch_size=N, ddim=use_ddim,
#                                              ddim_steps=ddim_steps, eta=ddim_eta,
#                                              unconditional_guidance_scale=unconditional_guidance_scale,
#                                              unconditional_conditioning=uc_full,
#                                              )
#             x_samples_cfg = self.decode_first_stage(samples_cfg)
#             log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"] = x_samples_cfg

#             # uc_cat_image = c_cat_image  # torch.zeros_like(c_cat_1)
#             # uc_full = {"c_concat": [uc_cat], "c_concat_image": [uc_cat_image], "c_crossattn": [uc_cross]}
#             # samples_cfg_image, _ = self.sample_log(cond={"c_concat": [c_cat_mask], "c_concat_image": [c_cat_image], "c_crossattn": [c]},
#             #                                  batch_size=N, ddim=use_ddim,
#             #                                  ddim_steps=ddim_steps, eta=ddim_eta,
#             #                                  unconditional_guidance_scale=unconditional_guidance_scale,
#             #                                  unconditional_conditioning=uc_full,
#             #                                  )
#             # x_samples_cfg_image = self.decode_first_stage(samples_cfg_image)
#             # log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_image"] = x_samples_cfg_image

#         return log

#     @torch.no_grad()
#     def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
#         ddim_sampler = DDIMSampler(self)
#         b, c, h, w = cond["c_concat"][0].shape # 這是原始圖片/遮罩的尺寸 (例如 512x512)
#         shape = (self.channels, h // 8, w // 8) #潛在空間的標準特徵圖尺寸
#         samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
#         return samples, intermediates

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         params = list(self.control_model.parameters())
#         if not self.sd_locked:
#             params += list(self.model.diffusion_model.output_blocks.parameters())
#             params += list(self.model.diffusion_model.out.parameters())
#         opt = torch.optim.AdamW(params, lr=lr)
#         return opt

#     def low_vram_shift(self, is_diffusing):
#         if is_diffusing:
#             self.model = self.model.cuda()
#             self.control_model = self.control_model.cuda() #學生/Mask 分枝
#             self.first_stage_model = self.first_stage_model.cpu()
#             self.cond_stage_model = self.cond_stage_model.cpu()
#         else:
#             self.model = self.model.cpu()
#             self.control_model = self.control_model.cpu()
#             self.first_stage_model = self.first_stage_model.cuda()
#             self.cond_stage_model = self.cond_stage_model.cuda()

#     def p_losses(self, x_start, cond, t, noise=None):

#         cond_mask = {}
#         cond_mask["c_crossattn"] = [cond["c_crossattn"][0]]
#         cond_mask["c_concat"] = [cond["c_concat_mask"][0]]

#         cond_image = {}
#         cond_image["c_crossattn"] = [cond["c_crossattn"][0]]
#         cond_image["c_concat"] = [cond["c_concat_mask"][0]]
#         cond_image["c_concat_image"] = [cond["c_concat_image"][0]]

#         weights_ones = torch.ones_like(t).to(x_start.device)
#         weights_thre = torch.where(t <= 200, torch.tensor(1), torch.tensor(0))

#         weights_mask = 1.0 * weights_ones  # Loss 0
#         weights_image = 1.0 * weights_ones  # Loss 1
#         weights_mask_2_image = 1.0 * weights_ones  # Loss 2
#         weights_mask_regularization = 1.0 * weights_thre  # Loss 3

#         noise = default(noise, lambda: torch.randn_like(x_start))
#         x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
#         model_output_mask = self.apply_model(x_noisy, t, cond_mask)

#         loss_dict = {}
#         prefix = 'train' if self.training else 'val'

#         if self.parameterization == "x0":
#             target = x_start
#         elif self.parameterization == "eps":
#             target = noise
#         elif self.parameterization == "v":
#             target = self.get_v(x_start, noise, t)
#         else:
#             raise NotImplementedError()

#         loss_base = self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
        
#         # 初始化 loss_simple，之後所有的強化 (Focus, clDice) 都加在它身上
#         loss_simple = loss_base.clone()
#         print(f"loss_simple_mask: {loss_simple.mean()}")
#         mask_for_attn = cond["c_concat_mask"][0].to(device=x_start.device, dtype=x_start.dtype) #字典中的鍵中的第一個值(硬體、精度對齊)
#         if mask_for_attn.max() > 1.0:
#             mask_for_attn = mask_for_attn / 255.0
#             mask_for_attn = (mask_for_attn > 0.5).float()
#         # 縮放遮罩尺寸以符合影像
#         if mask_for_attn.shape[-2:] != x_start.shape[-2:]: #如果遮罩與目前正在計算 Loss 的影像高度 和寬度不同
#             mask_for_attn = torch.nn.functional.interpolate( #最近鄰插值（縮放）遮罩的影像高度 和寬度
#                 mask_for_attn,
#                 size=x_start.shape[-2:],
#                 mode="nearest"
#             )
#         # 轉為單通道
#         if mask_for_attn.shape[1] != 1:
#             mask_for_attn = mask_for_attn.mean(dim=1, keepdim=True) #在第 1 個維度（也就是 Channel 維度）進行壓縮，轉單通道

#         # 定義模糊函數(空間注意力圖)
#         def gaussian_blur(x, sigma=2.0):
#             kernel_size = int(2 * ceil(2 * sigma) + 1)
#             # 建立高斯核 (這裡簡化使用內建卷積，確保在 GPU 上執行)
#             # 你也可以直接用 torchvision.transforms.functional.gaussian_blur
#             from torchvision.transforms import functional as TF
#             return TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

#         # 對原始 binary mask 先做模糊，再計算權重 map
#         smoothed_mask = gaussian_blur(mask_for_attn, sigma=1.5) 
        
#         # 建立權重矩陣：現在權重會從 (1 + focus_weight) 緩慢降到 1.0
#         attn_map = smoothed_mask * self.attn_focus_weight + 1.0
#         pixel_error = (model_output_mask - target)**2 #每個像素的平方誤差
#         loss_cvc_focus = (
#                         (pixel_error * attn_map).sum(dim=[1, 2, 3]) /
#                         (attn_map.sum(dim=[1, 2, 3]) * pixel_error.shape[1]).clamp_min(1e-6)
#                         )#如果該像素在遮罩上，它的誤差會乘以 6。如果該像素在背景，它的誤差維持不變（乘以 1）。
#         loss_cvc_focus = loss_cvc_focus.to(x_start.dtype)

#         prefix = 'train' if self.training else 'val'
#         loss_dict.update({f'{prefix}/loss_cvc_focus': loss_cvc_focus.mean()})
        
#         # 將此損失加入總和 (給予 0.5 的權重，可視實驗調整)
#         loss_simple = loss_simple + self.cvc_weight * loss_cvc_focus
#         print(f">>> Current CVC Weight: {self.cvc_weight}")
#         print(f"loss_cvc_focus: {loss_cvc_focus.mean():.6f}")
#         if weights_image.all():
#             model_output_image = self.apply_model(x_noisy, t, cond_image)
#             loss_simple_image = self.get_loss(model_output_image, target, mean=False).mean([1, 2, 3])
#             print(f"loss_simple_image: {loss_simple_image.mean()}")
#             loss_simple = loss_simple + weights_image * loss_simple_image

#         if weights_mask_2_image.all():
#             loss_simple_mask_2_image = self.get_loss(model_output_mask, model_output_image.detach(), mean=False).mean([1, 2, 3])
#             print(f"loss_simple_mask_2_image: {loss_simple_mask_2_image.mean()}")
#             loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image

#         if (self.global_step > (self.trainer.max_steps * 1 / 3)) and weights_mask_regularization.any(): # Done!
#             recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
#             noise_image_2_mask = default(noise, lambda: torch.randn_like(recon_output_image))
#             x_noisy_mask_recon = self.q_sample(x_start=recon_output_image, t=t, noise=noise_image_2_mask)

#             model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask)
#             loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
#             print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean()}")
#             loss_simple = loss_simple + weights_mask_regularization * loss_simple_mask_regularization
#         loss_dict.update({f'{prefix}/loss_simple_base': loss_base.mean()})
        
#         # --- [重點 2]：引導損失組合 (用於引導模型長出導管) ---
#         # 將原本累積在 loss_simple 上的東西保持，但確保權重不要過大
#         loss_dict.update({f'{prefix}/loss_simple_total_guided': loss_simple.mean()})

#         # --- [重點 3]：防綠防火牆邏輯 (真正的修正處) ---
#         logvar_t = self.logvar[t].to(self.device)
        
#         # 1. 計算加權後的 Simple Loss (包含你所有的引導強化)
#         weighted_simple = (loss_simple / torch.exp(logvar_t) + logvar_t).mean()
        
#         # 2. 計算加權後的 VLB (必須使用純淨的 loss_base，絕對不能包含 Focus 或 Reg)
#         # 這是為了確保 Latent Space 的穩定，不會因為 Focus 權重太強而炸掉
#         weighted_vlb = (self.lvlb_weights[t] * loss_base).mean() 
        
#         # --- [重點 4]：最終 Loss 合併 ---
#         # 依照 Latent Diffusion 論文標準公式合併
#         total_final_loss = self.l_simple_weight * weighted_simple + self.original_elbo_weight * weighted_vlb

#         # 更新字典記錄 (方便你在 Tensorboard / WandB 監看)
#         loss_dict.update({f'{prefix}/loss_vlb': weighted_vlb})
#         loss_dict.update({f'{prefix}/loss': total_final_loss})

#         if self.learn_logvar:
#             loss_dict.update({f'{prefix}/loss_gamma': weighted_simple})
#             loss_dict.update({'logvar': self.logvar.data.mean()})

#         return total_final_loss, loss_dict

# #沒損失沒底圖
# import einops
# import torch
# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F

# from ldm.modules.diffusionmodules.util import (
#     conv_nd,
#     linear,
#     zero_module,
#     timestep_embedding,
# )

# from einops import rearrange, repeat
# from torchvision.utils import make_grid
# from ldm.modules.attention import SpatialTransformer
# from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# from ldm.models.diffusion.ddpm import LatentDiffusion
# from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
# from ldm.models.diffusion.ddim import DDIMSampler

# import copy
# from cldm.dhi import FeatureExtractor


# class ControlledUnetModel(UNetModel):
#     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
#         hs = []
#         with torch.no_grad():
#             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#             emb = self.time_embed(t_emb)
#             h = x.type(self.dtype)
#             for module in self.input_blocks:
#                 h = module(h, emb, context)
#                 hs.append(h)
#             h = self.middle_block(h, emb, context)

#         if control is not None:
#             h += control.pop()

#         for i, module in enumerate(self.output_blocks):
#             if only_mid_control or control is None:
#                 h = torch.cat([h, hs.pop()], dim=1)
#             else:
#                 h = torch.cat([h, hs.pop() + control.pop()], dim=1)
#             h = module(h, emb, context)

#         h = h.type(x.dtype)
#         return self.out(h)


# class ControlNet(nn.Module):
#     def __init__(
#             self,
#             image_size,
#             in_channels,
#             model_channels,
#             hint_channels,
#             num_res_blocks,
#             attention_resolutions,
#             dropout=0,
#             channel_mult=(1, 2, 4, 8),
#             conv_resample=True,
#             dims=2,
#             use_checkpoint=False,
#             use_fp16=False,
#             num_heads=-1,
#             num_head_channels=-1,
#             num_heads_upsample=-1,
#             use_scale_shift_norm=False,
#             resblock_updown=False,
#             use_new_attention_order=False,
#             use_spatial_transformer=False,  # custom transformer support
#             transformer_depth=1,  # custom transformer support
#             context_dim=None,  # custom transformer support
#             n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
#             legacy=True,
#             disable_self_attentions=None,
#             num_attention_blocks=None,
#             disable_middle_self_attn=False,
#             use_linear_in_transformer=False,
#     ):
#         super().__init__()
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         if num_heads == -1:
#             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

#         if num_head_channels == -1:
#             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

#         self.dims = dims
#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         if isinstance(num_res_blocks, int):
#             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
#         else:
#             if len(num_res_blocks) != len(channel_mult):
#                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
#                                  "as a list/tuple (per-level) with the same length as channel_mult")
#             self.num_res_blocks = num_res_blocks
#         if disable_self_attentions is not None:
#             # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
#             assert len(disable_self_attentions) == len(channel_mult)
#         if num_attention_blocks is not None:
#             assert len(num_attention_blocks) == len(self.num_res_blocks)
#             assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
#             print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
#                   f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
#                   f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
#                   f"attention will still not be set.")

#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#         self.predict_codebook_ids = n_embed is not None

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
#                 )
#             ]
#         )
#         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

#         # self.input_hint_block = TimestepEmbedSequential(
#         #     conv_nd(dims, hint_channels, 16, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 16, 16, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 16, 32, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 32, 32, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 32, 96, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 96, 96, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 96, 256, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         # )

#         self.input_hint_block = TimestepEmbedSequential(
#             FeatureExtractor(hint_channels),
#             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         )

#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for nr in range(self.num_res_blocks[level]):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         # num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     if exists(disable_self_attentions):
#                         disabled_sa = disable_self_attentions[level]
#                     else:
#                         disabled_sa = False

#                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
#                         layers.append(
#                             AttentionBlock(
#                                 ch,
#                                 use_checkpoint=use_checkpoint,
#                                 num_heads=num_heads,
#                                 num_head_channels=dim_head,
#                                 use_new_attention_order=use_new_attention_order,
#                             ) if not use_spatial_transformer else SpatialTransformer(
#                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
#                                 use_checkpoint=use_checkpoint
#                             )
#                         )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             # num_heads = 1
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
#                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
#                 use_checkpoint=use_checkpoint
#             ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self.middle_block_out = self.make_zero_conv(ch)
#         self._feature_size += ch

#     def make_zero_conv(self, channels):
#         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

#     def forward(self, x, hint, timesteps, context, **kwargs):
#         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#         emb = self.time_embed(t_emb)

#         guided_hint = self.input_hint_block(hint, emb, context)

#         outs = []

#         h = x.type(self.dtype)
#         for module, zero_conv in zip(self.input_blocks, self.zero_convs):
#             if guided_hint is not None:
#                 h = module(h, emb, context)
#                 h += guided_hint
#                 guided_hint = None
#             else:
#                 h = module(h, emb, context)
#             outs.append(zero_conv(h, emb, context))

#         h = self.middle_block(h, emb, context)
#         outs.append(self.middle_block_out(h, emb, context))

#         return outs


# class ControlLDM(LatentDiffusion):

#     def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.control_model = instantiate_from_config(control_stage_config)
#         self.control_key = control_key
#         self.only_mid_control = only_mid_control
#         self.control_scales = [1.0] * 13 #ControlNet 提取出的「特徵組數」，用來對應並注入到 UNet 的特定層級中。

#     @torch.no_grad()
#     def get_input(self, batch, k, bs=None, *args, **kwargs):
#         x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
#         control_mask = batch[self.control_key]
#         if bs is not None:
#             control_mask = control_mask[:bs]
#         control_mask = control_mask.to(self.device)
#         control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
#         control_mask = control_mask.to(memory_format=torch.contiguous_format).float()

#         control_image = (batch["jpg"] + 1.0) / 2.0
#         if bs is not None:
#             control_image = control_image[:bs]
#         control_image = control_image.to(self.device)
#         control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
#         control_image = control_image.to(memory_format=torch.contiguous_format).float()

#         return x, dict(c_crossattn=[c], c_concat_mask=[control_mask], c_concat_image=[control_image])

#     def apply_model(self, x_noisy, t, cond, *args, **kwargs):
#         assert isinstance(cond, dict)
#         diffusion_model = self.model.diffusion_model

#         cond_txt = torch.cat(cond['c_crossattn'], 1)

#         if cond['c_concat'] is None:
#             eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
#         else:
#             if 'c_concat_image' in cond:
#                 control_model_mask = copy.deepcopy(self.control_model).requires_grad_(False)
#                 diffusion_model_image = copy.deepcopy(diffusion_model)         
#                 control_weights_mask = 1.0

# # --- 這裡改用 try-except 來徹底避開 Lightning 的屬性存取報錯 ---
#                 try:
#                     # 只有在正式 Trainer 環境下，這兩行才能執行成功
#                     t_max = self.trainer.max_steps
#                     g_step = self.global_step
#                     control_weights_image = 1.0 * g_step / t_max
#                 except (AttributeError, RuntimeError, Exception):
#                     # 如果在推理腳本中，存取上述變數會拋出異常，則直接給預設值
#                     control_weights_image = 1.0
#                 # ---------------------------------------------------------
#                 control_image = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat_image'], 1), timesteps=t, context=cond_txt)
#                 control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]
#                 with torch.no_grad():
#                     control_mask = control_model_mask(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
#                     control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]
#                 control = [control_weights_mask * c_mask.detach() + control_weights_image * c_image for c_mask, c_image in zip(control_mask, control_image)]
#                 eps = diffusion_model_image(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
#             else:
#                 control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
#                 control = [c * scale for c, scale in zip(control, self.control_scales)]
#                 eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

#         return eps

#     @torch.no_grad()
#     def get_unconditional_conditioning(self, N):
#         return self.get_learned_conditioning([""] * N)

#     @torch.no_grad()
#     def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
#                    quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
#                    plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
#                    use_ema_scope=True,
#                    **kwargs):
#         use_ddim = ddim_steps is not None

#         log = dict()
#         z, c = self.get_input(batch, self.first_stage_key, bs=N)
#         c_cat_mask, c_cat_image, c = c["c_concat_mask"][0][:N], c["c_concat_image"][0][:N], c["c_crossattn"][0][:N]
#         N = min(z.shape[0], N)
#         n_row = min(z.shape[0], n_row)
#         log["control_mask"] = c_cat_mask * 2.0 - 1.0
#         log["control_image"] = c_cat_image * 2.0 - 1.0
#         log["conditioning"] = log_txt_as_img((384, 384), batch[self.cond_stage_key], size=16)

#         if plot_diffusion_rows:
#             # get diffusion row
#             diffusion_row = list()
#             z_start = z[:n_row]
#             for t in range(self.num_timesteps):
#                 if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
#                     t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
#                     t = t.to(self.device).long()
#                     noise = torch.randn_like(z_start)
#                     z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
#                     diffusion_row.append(self.decode_first_stage(z_noisy))

#             diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
#             diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
#             diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
#             diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
#             log["diffusion_row"] = diffusion_grid

#         if sample:
#             # get denoise row
#             samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c]},
#                                                      batch_size=N, ddim=use_ddim,
#                                                      ddim_steps=ddim_steps, eta=ddim_eta)
#             x_samples = self.decode_first_stage(samples)
#             log["samples"] = x_samples
#             if plot_denoise_rows:
#                 denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
#                 log["denoise_row"] = denoise_grid

#         if unconditional_guidance_scale > 1.0:
#             uc_cross = self.get_unconditional_conditioning(N)
#             uc_cat = c_cat_mask  # torch.zeros_like(c_cat)
#             uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
#             samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat_mask], "c_crossattn": [c]},
#                                              batch_size=N, ddim=use_ddim,
#                                              ddim_steps=ddim_steps, eta=ddim_eta,
#                                              unconditional_guidance_scale=unconditional_guidance_scale,
#                                              unconditional_conditioning=uc_full,
#                                              )
#             x_samples_cfg = self.decode_first_stage(samples_cfg)
#             log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"] = x_samples_cfg

#             # uc_cat_image = c_cat_image  # torch.zeros_like(c_cat_1)
#             # uc_full = {"c_concat": [uc_cat], "c_concat_image": [uc_cat_image], "c_crossattn": [uc_cross]}
#             # samples_cfg_image, _ = self.sample_log(cond={"c_concat": [c_cat_mask], "c_concat_image": [c_cat_image], "c_crossattn": [c]},
#             #                                  batch_size=N, ddim=use_ddim,
#             #                                  ddim_steps=ddim_steps, eta=ddim_eta,
#             #                                  unconditional_guidance_scale=unconditional_guidance_scale,
#             #                                  unconditional_conditioning=uc_full,
#             #                                  )
#             # x_samples_cfg_image = self.decode_first_stage(samples_cfg_image)
#             # log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_image"] = x_samples_cfg_image

#         return log

#     @torch.no_grad()
#     def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
#         ddim_sampler = DDIMSampler(self)
#         b, c, h, w = cond["c_concat"][0].shape
#         shape = (self.channels, h // 8, w // 8)
#         samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
#         return samples, intermediates

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         params = list(self.control_model.parameters())
#         if not self.sd_locked:
#             params += list(self.model.diffusion_model.output_blocks.parameters())
#             params += list(self.model.diffusion_model.out.parameters())
#         opt = torch.optim.AdamW(params, lr=lr)
#         return opt

#     def low_vram_shift(self, is_diffusing):
#         if is_diffusing:
#             self.model = self.model.cuda()
#             self.control_model = self.control_model.cuda()
#             self.first_stage_model = self.first_stage_model.cpu()
#             self.cond_stage_model = self.cond_stage_model.cpu()
#         else:
#             self.model = self.model.cpu()
#             self.control_model = self.control_model.cpu()
#             self.first_stage_model = self.first_stage_model.cuda()
#             self.cond_stage_model = self.cond_stage_model.cuda()

#     def p_losses(self, x_start, cond, t, noise=None):

#         cond_mask = {}
#         cond_mask["c_crossattn"] = [cond["c_crossattn"][0]]
#         cond_mask["c_concat"] = [cond["c_concat_mask"][0]]

#         cond_image = {}
#         cond_image["c_crossattn"] = [cond["c_crossattn"][0]]
#         cond_image["c_concat"] = [cond["c_concat_mask"][0]]
#         cond_image["c_concat_image"] = [cond["c_concat_image"][0]]

#         weights_ones = torch.ones_like(t).to(x_start.device)
#         weights_thre = torch.where(t <= 200, torch.tensor(1), torch.tensor(0))

#         weights_mask = 1.0 * weights_ones  # Loss 0
#         weights_image = 1.0 * weights_ones  # Loss 1
#         weights_mask_2_image = 1.0 * weights_ones  # Loss 2
#         weights_mask_regularization = 1.0 * weights_thre  # Loss 3

#         noise = default(noise, lambda: torch.randn_like(x_start))
#         x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
#         model_output_mask = self.apply_model(x_noisy, t, cond_mask)

#         loss_dict = {}
#         prefix = 'train' if self.training else 'val'

#         if self.parameterization == "x0":
#             target = x_start
#         elif self.parameterization == "eps":
#             target = noise
#         elif self.parameterization == "v":
#             target = self.get_v(x_start, noise, t)
#         else:
#             raise NotImplementedError()

#         loss_simple = weights_mask * self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
#         print(f"loss_simple_mask: {loss_simple.mean()}")

#         if weights_image.all():
#             model_output_image = self.apply_model(x_noisy, t, cond_image)
#             loss_simple_image = self.get_loss(model_output_image, target, mean=False).mean([1, 2, 3])
#             print(f"loss_simple_image: {loss_simple_image.mean()}")
#             loss_simple = loss_simple + weights_image * loss_simple_image

#         if weights_mask_2_image.all():
#             loss_simple_mask_2_image = self.get_loss(model_output_mask, model_output_image.detach(), mean=False).mean([1, 2, 3])
#             print(f"loss_simple_mask_2_image: {loss_simple_mask_2_image.mean()}")
#             loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image

#         if (self.global_step > (self.trainer.max_steps * 1 / 3)) and weights_mask_regularization.any(): # Done!
#             recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
#             noise_image_2_mask = default(noise, lambda: torch.randn_like(recon_output_image))
#             x_noisy_mask_recon = self.q_sample(x_start=recon_output_image, t=t, noise=noise_image_2_mask)

#             model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask)
#             loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
#             print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean()}")
#             loss_simple = loss_simple + weights_mask_regularization * loss_simple_mask_regularization

#         loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

#         logvar_t = self.logvar[t].to(self.device)
#         loss = loss_simple / torch.exp(logvar_t) + logvar_t
#         # loss = loss_simple / torch.exp(self.logvar) + self.logvar
#         if self.learn_logvar:
#             loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
#             loss_dict.update({'logvar': self.logvar.data.mean()})

#         loss = self.l_simple_weight * loss.mean()

#         loss_vlb = loss_simple

#         loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
#         loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
#         loss += (self.original_elbo_weight * loss_vlb)
#         loss_dict.update({f'{prefix}/loss': loss})

#         return loss, loss_dict


# #有底圖無損失
# import einops
# import torch
# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F

# from ldm.modules.diffusionmodules.util import (
#     conv_nd,
#     linear,
#     zero_module,
#     timestep_embedding,
# )

# from einops import rearrange, repeat
# from torchvision.utils import make_grid
# from ldm.modules.attention import SpatialTransformer
# from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
# from ldm.models.diffusion.ddpm import LatentDiffusion
# from ldm.util import log_txt_as_img, exists, instantiate_from_config, default
# from ldm.models.diffusion.ddim import DDIMSampler
# import torchvision.transforms.functional as TF
# import copy
# from cldm.dhi import FeatureExtractor


# class ControlledUnetModel(UNetModel):
#     def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
#         hs = []
#         with torch.no_grad():
#             t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#             emb = self.time_embed(t_emb)
#             h = x.type(self.dtype)
#             for module in self.input_blocks:
#                 h = module(h, emb, context)
#                 hs.append(h)
#             h = self.middle_block(h, emb, context)

#         if control is not None:
#             h += control.pop()

#         for i, module in enumerate(self.output_blocks):
#             if only_mid_control or control is None:
#                 h = torch.cat([h, hs.pop()], dim=1)
#             else:
#                 h = torch.cat([h, hs.pop() + control.pop()], dim=1)
#             h = module(h, emb, context)

#         h = h.type(x.dtype)
#         return self.out(h)


# class ControlNet(nn.Module):
#     def __init__(
#             self,
#             image_size,
#             in_channels,
#             model_channels,
#             hint_channels,
#             num_res_blocks,
#             attention_resolutions,
#             dropout=0,
#             channel_mult=(1, 2, 4, 8),
#             conv_resample=True,
#             dims=2,
#             use_checkpoint=False,
#             use_fp16=False,
#             num_heads=-1,
#             num_head_channels=-1,
#             num_heads_upsample=-1,
#             use_scale_shift_norm=False,
#             resblock_updown=False,
#             use_new_attention_order=False,
#             use_spatial_transformer=False,  # custom transformer support
#             transformer_depth=1,  # custom transformer support
#             context_dim=None,  # custom transformer support
#             n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
#             legacy=True,
#             disable_self_attentions=None,
#             num_attention_blocks=None,
#             disable_middle_self_attn=False,
#             use_linear_in_transformer=False,
#     ):
#         super().__init__()
#         if use_spatial_transformer:
#             assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

#         if context_dim is not None:
#             assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
#             from omegaconf.listconfig import ListConfig
#             if type(context_dim) == ListConfig:
#                 context_dim = list(context_dim)

#         if num_heads_upsample == -1:
#             num_heads_upsample = num_heads

#         if num_heads == -1:
#             assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

#         if num_head_channels == -1:
#             assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

#         self.dims = dims
#         self.image_size = image_size
#         self.in_channels = in_channels
#         self.model_channels = model_channels
#         if isinstance(num_res_blocks, int):
#             self.num_res_blocks = len(channel_mult) * [num_res_blocks]
#         else:
#             if len(num_res_blocks) != len(channel_mult):
#                 raise ValueError("provide num_res_blocks either as an int (globally constant) or "
#                                  "as a list/tuple (per-level) with the same length as channel_mult")
#             self.num_res_blocks = num_res_blocks
#         if disable_self_attentions is not None:
#             # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
#             assert len(disable_self_attentions) == len(channel_mult)
#         if num_attention_blocks is not None:
#             assert len(num_attention_blocks) == len(self.num_res_blocks)
#             assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
#             print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
#                   f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
#                   f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
#                   f"attention will still not be set.")

#         self.attention_resolutions = attention_resolutions
#         self.dropout = dropout
#         self.channel_mult = channel_mult
#         self.conv_resample = conv_resample
#         self.use_checkpoint = use_checkpoint
#         self.dtype = th.float16 if use_fp16 else th.float32
#         self.num_heads = num_heads
#         self.num_head_channels = num_head_channels
#         self.num_heads_upsample = num_heads_upsample
#         self.predict_codebook_ids = n_embed is not None

#         time_embed_dim = model_channels * 4
#         self.time_embed = nn.Sequential(
#             linear(model_channels, time_embed_dim),
#             nn.SiLU(),
#             linear(time_embed_dim, time_embed_dim),
#         )

#         self.input_blocks = nn.ModuleList(
#             [
#                 TimestepEmbedSequential(
#                     conv_nd(dims, in_channels, model_channels, 3, padding=1)
#                 )
#             ]
#         )
#         self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

#         # self.input_hint_block = TimestepEmbedSequential(
#         #     conv_nd(dims, hint_channels, 16, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 16, 16, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 16, 32, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 32, 32, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 32, 96, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 96, 96, 3, padding=1),
#         #     nn.SiLU(),
#         #     conv_nd(dims, 96, 256, 3, padding=1, stride=2),
#         #     nn.SiLU(),
#         #     zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         # )

#         self.input_hint_block = TimestepEmbedSequential(
#             FeatureExtractor(hint_channels),
#             zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
#         )

#         self._feature_size = model_channels
#         input_block_chans = [model_channels]
#         ch = model_channels
#         ds = 1
#         for level, mult in enumerate(channel_mult):
#             for nr in range(self.num_res_blocks[level]):
#                 layers = [
#                     ResBlock(
#                         ch,
#                         time_embed_dim,
#                         dropout,
#                         out_channels=mult * model_channels,
#                         dims=dims,
#                         use_checkpoint=use_checkpoint,
#                         use_scale_shift_norm=use_scale_shift_norm,
#                     )
#                 ]
#                 ch = mult * model_channels
#                 if ds in attention_resolutions:
#                     if num_head_channels == -1:
#                         dim_head = ch // num_heads
#                     else:
#                         num_heads = ch // num_head_channels
#                         dim_head = num_head_channels
#                     if legacy:
#                         # num_heads = 1
#                         dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#                     if exists(disable_self_attentions):
#                         disabled_sa = disable_self_attentions[level]
#                     else:
#                         disabled_sa = False

#                     if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
#                         layers.append(
#                             AttentionBlock(
#                                 ch,
#                                 use_checkpoint=use_checkpoint,
#                                 num_heads=num_heads,
#                                 num_head_channels=dim_head,
#                                 use_new_attention_order=use_new_attention_order,
#                             ) if not use_spatial_transformer else SpatialTransformer(
#                                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                                 disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
#                                 use_checkpoint=use_checkpoint
#                             )
#                         )
#                 self.input_blocks.append(TimestepEmbedSequential(*layers))
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 self._feature_size += ch
#                 input_block_chans.append(ch)
#             if level != len(channel_mult) - 1:
#                 out_ch = ch
#                 self.input_blocks.append(
#                     TimestepEmbedSequential(
#                         ResBlock(
#                             ch,
#                             time_embed_dim,
#                             dropout,
#                             out_channels=out_ch,
#                             dims=dims,
#                             use_checkpoint=use_checkpoint,
#                             use_scale_shift_norm=use_scale_shift_norm,
#                             down=True,
#                         )
#                         if resblock_updown
#                         else Downsample(
#                             ch, conv_resample, dims=dims, out_channels=out_ch
#                         )
#                     )
#                 )
#                 ch = out_ch
#                 input_block_chans.append(ch)
#                 self.zero_convs.append(self.make_zero_conv(ch))
#                 ds *= 2
#                 self._feature_size += ch

#         if num_head_channels == -1:
#             dim_head = ch // num_heads
#         else:
#             num_heads = ch // num_head_channels
#             dim_head = num_head_channels
#         if legacy:
#             # num_heads = 1
#             dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
#         self.middle_block = TimestepEmbedSequential(
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#             AttentionBlock(
#                 ch,
#                 use_checkpoint=use_checkpoint,
#                 num_heads=num_heads,
#                 num_head_channels=dim_head,
#                 use_new_attention_order=use_new_attention_order,
#             ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
#                 ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
#                 disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
#                 use_checkpoint=use_checkpoint
#             ),
#             ResBlock(
#                 ch,
#                 time_embed_dim,
#                 dropout,
#                 dims=dims,
#                 use_checkpoint=use_checkpoint,
#                 use_scale_shift_norm=use_scale_shift_norm,
#             ),
#         )
#         self.middle_block_out = self.make_zero_conv(ch)
#         self._feature_size += ch

#     def make_zero_conv(self, channels):
#         return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))

#     def forward(self, x, hint, timesteps, context, **kwargs):
#         t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
#         emb = self.time_embed(t_emb)

#         guided_hint = self.input_hint_block(hint, emb, context)

#         outs = []

#         h = x.type(self.dtype)
#         for module, zero_conv in zip(self.input_blocks, self.zero_convs):
#             if guided_hint is not None:
#                 h = module(h, emb, context)
#                 h += guided_hint
#                 guided_hint = None
#             else:
#                 h = module(h, emb, context)
#             outs.append(zero_conv(h, emb, context))

#         h = self.middle_block(h, emb, context)
#         outs.append(self.middle_block_out(h, emb, context))

#         return outs


# class ControlLDM(LatentDiffusion):

#     def __init__(self, control_stage_config, control_key, only_mid_control, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # 原有的 Mask 控制模型
#         self.control_model = instantiate_from_config(control_stage_config)
        
#         # 新增：獨立的影像控制模型 (複製一份相同的架構)
#         self.control_model_image = instantiate_from_config(control_stage_config)
        
#         self.control_key = control_key
#         self.only_mid_control = only_mid_control
#         self.control_scales = [1.0] * 13 #ControlNet 提取出的「特徵組數」，用來對應並注入到 UNet 的特定層級中。
#         self.control_alpha = nn.Parameter(torch.tensor(0.1))
#     @torch.no_grad()
#     def get_input(self, batch, k, bs=None, *args, **kwargs):
#         x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
#         control_mask = batch[self.control_key]
#         if bs is not None:
#             control_mask = control_mask[:bs]
#         control_mask = control_mask.to(self.device)
#         control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
#         if self.training: # 只有訓練時模糊，或者推論也模糊以保持一致
#             control_mask = TF.gaussian_blur(control_mask, kernel_size=[5, 5], sigma=[1.0, 1.0])
#         if self.training: # 建議只在訓練階段加入隨機性
#             # 產生一個 0.6 到 1.0 之間的隨機數
#             random_opacity = torch.rand(control_mask.shape[0], 1, 1, 1).to(self.device) * 0.4 + 0.6
#             control_mask = control_mask * random_opacity
#         control_mask = control_mask.to(memory_format=torch.contiguous_format).float()

#         control_image = (batch["jpg"] + 1.0) / 2.0 # 讀取 jpg (底圖)
#         if bs is not None:
#             control_image = control_image[:bs]
#         control_image = control_image.to(self.device)
#         control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
#         control_image = control_image.to(memory_format=torch.contiguous_format).float()

#         return x, dict(c_crossattn=[c], c_concat_mask=[control_mask], c_concat_image=[control_image])

#     def apply_model(self, x_noisy, t, cond, *args, **kwargs):
#         assert isinstance(cond, dict)
#         diffusion_model = self.model.diffusion_model
        
#         # 取得 Text Conditioning (Cross-Attention)
#         cond_txt = torch.cat(cond['c_crossattn'], 1)

#         # 初始化控制變數
#         control_mask = None
#         control_image = None
#         control = None

#         # 1. Mask 分枝 (使用原本的 control_model)
#         # 對應 p_losses 中的 cond_mask["c_concat"]
#         if 'c_concat_mask' in cond:
#             # hint 取自 c_concat (通常是黑白 Mask 或線稿)
#             mask_hint = torch.cat(cond['c_concat_mask'], 1)
#             control_mask = self.control_model(
#                 x=x_noisy, 
#                 hint=mask_hint, 
#                 timesteps=t, 
#                 context=cond_txt
#             )
#             # 套用控制強度控制 (Control Scales)
#             control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]

#         # 2. Image 分枝 (使用獨立的 control_model_image)
#         # 對應 p_losses 中的 cond_image["c_concat_image"]
#         if 'c_concat_image' in cond:
#             # hint 取自 c_concat_image (彩色影像)
#             image_hint = torch.cat(cond['c_concat_image'], 1)
#             control_image = self.control_model_image(
#                 x=x_noisy, 
#                 hint=image_hint, 
#                 timesteps=t, 
#                 context=cond_txt
#             )
#             # 影像分枝同樣套用 control_scales
#             control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]

#         # 3. 融合邏輯 (Fusion Logic)
#         if control_mask is not None and control_image is not None:
#             # 兩者皆有時，使用可學習的 alpha 進行融合
#             # 使用 clamp 確保 alpha 在 0~1 之間，避免數值爆炸
#             alpha = torch.clamp(self.control_alpha, 0.0, 1.0)
            
#             # 融合公式：Mask 特徵 + alpha * Image 特徵
#             control = [(1.0 - alpha) * c_m + alpha * c_i for c_m, c_i in zip(control_mask, control_image)]
            
#         elif control_mask is not None:
#             # 只有 Mask 時 (例如單獨訓練 Mask 分枝或推論時)
#             control = control_mask
            
#         elif control_image is not None:
#             # 只有 Image 時 (如果你的邏輯允許單獨影像控制)
#             control = control_image

#         # 4. 將最終融合後的 control 丟入 UNet
#         eps = diffusion_model(
#             x=x_noisy, 
#             timesteps=t, 
#             context=cond_txt, 
#             control=control, 
#             only_mid_control=self.only_mid_control
#         )
        
#         return eps

#     @torch.no_grad()
#     def get_unconditional_conditioning(self, N):
#         return self.get_learned_conditioning([""] * N)

#     @torch.no_grad()
#     def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
#                    quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
#                    plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
#                    use_ema_scope=True,
#                    **kwargs):
#         use_ddim = ddim_steps is not None

#         log = dict()
#         z, c = self.get_input(batch, self.first_stage_key, bs=N)
#         c_cat_mask, c_cat_image, c = c["c_concat_mask"][0][:N], c["c_concat_image"][0][:N], c["c_crossattn"][0][:N]
#         N = min(z.shape[0], N)
#         n_row = min(z.shape[0], n_row)
#         log["control_mask"] = c_cat_mask * 2.0 - 1.0
#         log["control_image"] = c_cat_image * 2.0 - 1.0
#         log["conditioning"] = log_txt_as_img((384, 384), batch[self.cond_stage_key], size=16)

#         if plot_diffusion_rows:
#             # get diffusion row
#             diffusion_row = list()
#             z_start = z[:n_row]
#             for t in range(self.num_timesteps):
#                 if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
#                     t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
#                     t = t.to(self.device).long()
#                     noise = torch.randn_like(z_start)
#                     z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
#                     diffusion_row.append(self.decode_first_stage(z_noisy))

#             diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
#             diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
#             diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
#             diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
#             log["diffusion_row"] = diffusion_grid

#         if sample:
#             # get denoise row
#             samples, z_denoise_row = self.sample_log(
#                 cond={
#                     "c_concat_mask": [c_cat_mask], 
#                         "c_concat_image": [c_cat_image], # 新增這一行
#                     "c_crossattn": [c]
#             },
#                 batch_size=N, ddim=use_ddim,
#                 ddim_steps=ddim_steps, eta=ddim_eta
#             )
#             x_samples = self.decode_first_stage(samples)
#             log["samples"] = x_samples
#             if plot_denoise_rows:
#                 denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
#                 log["denoise_row"] = denoise_grid

#         if unconditional_guidance_scale > 1.0:
#             uc_cross = self.get_unconditional_conditioning(N)
#             uc_cat_mask = torch.zeros_like(c_cat_mask) # 建議用 0，代表無條件
#             uc_cat_image = torch.zeros_like(c_cat_image)
            
#             # 這裡的 key 必須與 apply_model 內的名稱一致
#             uc_full = {
#                 "c_concat_mask": [uc_cat_mask], 
#                 "c_concat_image": [uc_cat_image], 
#                 "c_crossattn": [uc_cross]
#             }
#             samples_cfg, _ = self.sample_log(
#                         cond={
#                             "c_concat_mask": [c_cat_mask], 
#                             "c_concat_image": [c_cat_image], # 必須加上
#                             "c_crossattn": [c]
#                         },
#                         batch_size=N, ddim=use_ddim,
#                         ddim_steps=ddim_steps, eta=ddim_eta,
#                         unconditional_guidance_scale=unconditional_guidance_scale,
#                         unconditional_conditioning=uc_full,
#                     )
#             x_samples_cfg = self.decode_first_stage(samples_cfg)
#             log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"] = x_samples_cfg

#             uc_cat_image = c_cat_image  # torch.zeros_like(c_cat_1)
#             uc_full = {"c_concat_mask": [uc_cat_mask], "c_concat_image": [uc_cat_image], "c_crossattn": [uc_cross]}
#             samples_cfg_image, _ = self.sample_log(cond={"c_concat_mask": [c_cat_mask], "c_concat_image": [c_cat_image], "c_crossattn": [c]},
#                                              batch_size=N, ddim=use_ddim,
#                                              ddim_steps=ddim_steps, eta=ddim_eta,
#                                              unconditional_guidance_scale=unconditional_guidance_scale,
#                                              unconditional_conditioning=uc_full,
#                                              )
#             x_samples_cfg_image = self.decode_first_stage(samples_cfg_image)
#             log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_image"] = x_samples_cfg_image

#         return log

#     @torch.no_grad()
#     def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
#         ddim_sampler = DDIMSampler(self)
#         b, c, h, w = cond["c_concat_mask"][0].shape
#         shape = (self.channels, h // 8, w // 8)
#         samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
#         return samples, intermediates

#     def configure_optimizers(self): 
#         lr = self.learning_rate
    
#     # --- 凍結階段 ---
#     # 1. 凍結影像控制分支 (因為影像結構通常已經很強)
#         for param in self.control_model_image.parameters():
#             param.requires_grad = False
        
#     # 2. 確保 Mask 分支是開啟訓練的
#         for param in self.control_model.parameters():
#             param.requires_grad = True
        
#     # 3. 確保融合參數 alpha 是開啟訓練的
#         self.control_alpha.requires_grad = True
    
#     # --- 參數收集 ---
#     # 只把需要訓練的參數丟進優化器
#         params = list(self.control_model.parameters())
#         params += [self.control_alpha] 
    
#     # 如果 diffusion_model 本身沒有鎖定，也加入輸出層參數
#         if not self.sd_locked:
#         # 這裡也要確保這些參數的 requires_grad 是 True
#             for param in self.model.diffusion_model.output_blocks.parameters():
#                 param.requires_grad = True
#             for param in self.model.diffusion_model.out.parameters():
#                 param.requires_grad = True
            
#             params += list(self.model.diffusion_model.output_blocks.parameters())
#             params += list(self.model.diffusion_model.out.parameters())
    
#     # --- 建立優化器 ---
#     # 建議使用過濾後的列表，確保優化器不會去處理不需要梯度的參數
#         trainable_params = [p for p in params if p.requires_grad]
#         opt = torch.optim.AdamW(trainable_params, lr=lr)
    
#         return opt

#     def low_vram_shift(self, is_diffusing):
#         if is_diffusing:
#             self.model = self.model.cuda()
#             self.control_model = self.control_model.cuda()
#             self.control_model_image = self.control_model_image.cuda()
#             self.first_stage_model = self.first_stage_model.cpu()
#             self.cond_stage_model = self.cond_stage_model.cpu()
#         else:
#             self.model = self.model.cpu()
#             self.control_model = self.control_model.cpu()
#             self.control_model_image = self.control_model_image.cpu()
#             self.first_stage_model = self.first_stage_model.cuda()
#             self.cond_stage_model = self.cond_stage_model.cuda()

#     def p_losses(self, x_start, cond, t, noise=None):

#         # 準備只有 Mask 的條件
#         cond_mask = {
#             "c_crossattn": [cond["c_crossattn"][0]],
#             "c_concat_mask": [cond["c_concat_mask"][0]] 
#         }

#         # 準備包含 Mask + Image 的條件
#         cond_image = {
#             "c_crossattn": [cond["c_crossattn"][0]],
#             "c_concat_mask": [cond["c_concat_mask"][0]],
#             "c_concat_image": [cond["c_concat_image"][0]]
#         }

#         weights_ones = torch.ones_like(t).to(x_start.device)
#         weights_thre = torch.where(t <= 200, torch.tensor(1), torch.tensor(0))

#         weights_mask = 1.0 * weights_ones  # Loss 0
#         weights_image = 1.0 * weights_ones  # Loss 1
#         weights_mask_2_image = 1.0 * weights_ones  # Loss 2
#         weights_mask_regularization = 1.0 * weights_thre  # Loss 3

#         noise = default(noise, lambda: torch.randn_like(x_start))
#         x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
#         model_output_mask = self.apply_model(x_noisy, t, cond_mask)

#         loss_dict = {}
#         prefix = 'train' if self.training else 'val'

#         if self.parameterization == "x0":
#             target = x_start
#         elif self.parameterization == "eps":
#             target = noise
#         elif self.parameterization == "v":
#             target = self.get_v(x_start, noise, t)
#         else:
#             raise NotImplementedError()

#         loss_simple = weights_mask * self.get_loss(model_output_mask, target, mean=False).mean([1, 2, 3])
#         print(f"loss_simple_mask: {loss_simple.mean()}")

#         if weights_image.all():
#             model_output_image = self.apply_model(x_noisy, t, cond_image)
#             loss_simple_image = self.get_loss(model_output_image, target, mean=False).mean([1, 2, 3])
#             print(f"loss_simple_image: {loss_simple_image.mean()}")
#             loss_simple = loss_simple + weights_image * loss_simple_image

#         if weights_mask_2_image.all():
#             loss_simple_mask_2_image = self.get_loss(model_output_mask, model_output_image.detach(), mean=False).mean([1, 2, 3])
#             print(f"loss_simple_mask_2_image: {loss_simple_mask_2_image.mean()}")
#             loss_simple = loss_simple + weights_mask_2_image * loss_simple_mask_2_image

#         if (self.global_step > (self.trainer.max_steps * 1 / 3)) and weights_mask_regularization.any(): # Done!
#             recon_output_image = self.predict_start_from_noise(x_noisy, t=t, noise=model_output_image)
#             noise_image_2_mask = default(noise, lambda: torch.randn_like(recon_output_image))
#             x_noisy_mask_recon = self.q_sample(x_start=recon_output_image, t=t, noise=noise_image_2_mask)

#             model_output_mask_xt = self.apply_model(x_noisy_mask_recon.detach(), t, cond_mask)
#             loss_simple_mask_regularization = self.get_loss(model_output_mask_xt, noise_image_2_mask, mean=False).mean([1, 2, 3])
#             print(f"loss_simple_mask_regularization: {loss_simple_mask_regularization.mean()}")
#             loss_simple = loss_simple + weights_mask_regularization * loss_simple_mask_regularization

#         loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

#         logvar_t = self.logvar[t].to(self.device)
#         loss = loss_simple / torch.exp(logvar_t) + logvar_t
#         # loss = loss_simple / torch.exp(self.logvar) + self.logvar
#         if self.learn_logvar:
#             loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
#             loss_dict.update({'logvar': self.logvar.data.mean()})

#         loss = self.l_simple_weight * loss.mean()

#         loss_vlb = loss_simple

#         loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
#         loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
#         loss += (self.original_elbo_weight * loss_vlb)
#         loss_dict.update({f'{prefix}/loss': loss})

#         return loss, loss_dict

        #有底圖有損失
import einops
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

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
import torchvision.transforms.functional as TF
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
        self.control_model = instantiate_from_config(control_stage_config) # 原有的 Mask 控制模型
        # 這兩行使用的是同一個 control_stage_config，所以架構一模一樣
        # 新增：獨立的影像控制模型 (複製一份相同的架構)
        self.control_model_image = instantiate_from_config(control_stage_config)
        
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13 #ControlNet 提取出的「特徵組數」，用來對應並注入到 UNet 的特定層級中。
        self.control_alpha = nn.Parameter(torch.tensor(0.1)) #建立一個包含數值 0.1 的 PyTorch 張量，包裝成一個 nn.Parameter 物件，並在訓練過程中根據 Loss 的梯度來自動更新它。
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        print(f"\n[DEBUG] Batch Keys: {list(batch.keys())}\n") 
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control_mask = batch[self.control_key]  #去 batch 字典裡找那個叫做 self.control_key（例如 "hint"）的欄位。導管遮罩圖提取出來，轉換成 control_mask 張量。
        if bs is not None:
            control_mask = control_mask[:bs]
        control_mask = control_mask.to(self.device)
        control_mask = einops.rearrange(control_mask, 'b h w c -> b c h w')
        if self.training: # 只有訓練時模糊，或者推論也模糊以保持一致
            control_mask = TF.gaussian_blur(control_mask, kernel_size=[5, 5], sigma=[1.0, 1.0])
        if self.training: # 建議只在訓練階段加入隨機性
            # 產生一個 0.6 到 1.0 之間的隨機數
            random_opacity = torch.rand(control_mask.shape[0], 1, 1, 1).to(self.device) * 0.4 + 0.6
            control_mask = control_mask * random_opacity
        control_mask = control_mask.to(memory_format=torch.contiguous_format).float()

        control_image = (batch["jpg"] + 1.0) / 2.0 # 讀取 jpg (底圖)
        if bs is not None:
            control_image = control_image[:bs]
        control_image = control_image.to(self.device)
        control_image = einops.rearrange(control_image, 'b h w c -> b c h w')
        control_image = control_image.to(memory_format=torch.contiguous_format).float()

        return x, dict(c_crossattn=[c], c_concat_mask=[control_mask], c_concat_image=[control_image])

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model #UNet
        
        # 取得 Text Conditioning (Cross-Attention)
        cond_txt = torch.cat(cond['c_crossattn'], 1) #在第 1 維度（通道維度 Channel）進行串接。

        # 初始化控制變數
        control_mask = None
        control_image = None
        control = None

        # 1. Mask 分枝 (使用原本的 control_model)
        # 對應 p_losses 中的 cond_mask["c_concat"]
        if 'c_concat_mask' in cond:
            # hint 取自 c_concat (通常是黑白 Mask 或線稿)
            mask_hint = torch.cat(cond['c_concat_mask'], 1)
            control_mask = self.control_model( ##ControlNet 架構產生遮罩特徵
                x=x_noisy, 
                hint=mask_hint, 
                timesteps=t, 
                context=cond_txt
            )
            # 套用控制強度控制 (Control Scales)
            control_mask = [c * scale for c, scale in zip(control_mask, self.control_scales)]

        # 2. Image 分枝 (使用獨立的 control_model_image)
        # 對應 p_losses 中的 cond_image["c_concat_image"]
        if 'c_concat_image' in cond:
            # hint 取自 c_concat_image (彩色影像)
            image_hint = torch.cat(cond['c_concat_image'], 1)
            control_image = self.control_model_image( #ControlNet 架構產生影像特徵
                x=x_noisy, 
                hint=image_hint, 
                timesteps=t, 
                context=cond_txt
            )
            # 影像分枝同樣套用 control_scales
            control_image = [c * scale for c, scale in zip(control_image, self.control_scales)]

        # 3. 融合邏輯 (Fusion Logic) :兩者皆有時，使用可學習的 alpha 進行融合
        if control_mask is not None and control_image is not None:
            alpha = torch.sigmoid(self.control_alpha) #使用 sigmoid 的目的是將可學習參數 control_alpha 限制在 0 到 1 之間，確保融合比例的數值穩定性
            control = [(1.0 - alpha) * c_m + alpha * c_i for c_m, c_i in zip(control_mask, control_image)] #zip 會將「遮罩分枝」與「影像分枝」在相同層級的特徵圖配對在一起，遮罩分枝（Mask branch）提取的特徵與影像分枝（Image branch）提取的特徵加權平均
            
        elif control_mask is not None:
            # 只有 Mask 時 (例如單獨訓練 Mask 分枝或推論時)
            control = control_mask
            
        elif control_image is not None:
            # 只有 Image 時 (如果你的邏輯允許單獨影像控制)
            control = control_image

        # 4. 將最終融合後的 control 丟入 UNet
        eps = diffusion_model(
            x=x_noisy, 
            timesteps=t, 
            context=cond_txt, 
            control=control, 
            only_mid_control=self.only_mid_control
        )
        
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
            samples, z_denoise_row = self.sample_log(
                cond={
                    "c_concat_mask": [c_cat_mask], 
                        "c_concat_image": [c_cat_image], # 新增這一行
                    "c_crossattn": [c]
            },
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta
            )
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat_mask = torch.zeros_like(c_cat_mask) # 建議用 0，代表無條件
            uc_cat_image = torch.zeros_like(c_cat_image)
            
            # 這裡的 key 必須與 apply_model 內的名稱一致
            uc_full = {
                "c_concat_mask": [uc_cat_mask], 
                "c_concat_image": [uc_cat_image], 
                "c_crossattn": [uc_cross]
            }
            samples_cfg, _ = self.sample_log(
                        cond={
                            "c_concat_mask": [c_cat_mask], 
                            "c_concat_image": [c_cat_image], # 必須加上
                            "c_crossattn": [c]
                        },
                        batch_size=N, ddim=use_ddim,
                        ddim_steps=ddim_steps, eta=ddim_eta,
                        unconditional_guidance_scale=unconditional_guidance_scale,
                        unconditional_conditioning=uc_full,
                    )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_mask"] = x_samples_cfg

            uc_cat_image = c_cat_image  # torch.zeros_like(c_cat_1)
            uc_full = {"c_concat_mask": [uc_cat_mask], "c_concat_image": [uc_cat_image], "c_crossattn": [uc_cross]}
            samples_cfg_image, _ = self.sample_log(cond={"c_concat_mask": [c_cat_mask], "c_concat_image": [c_cat_image], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg_image = self.decode_first_stage(samples_cfg_image)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}_image"] = x_samples_cfg_image

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat_mask"][0].shape
        shape = (self.channels, h // 8, w // 8) #潛在空間的標準特徵圖尺寸
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs) # Sampler 根據這個 shape 產生一個隨機噪點張量 (Latent Noise)
        return samples, intermediates

    def configure_optimizers(self): 
        lr = self.learning_rate
    
    # --- 凍結階段 --凍結影像控制分支 (因為影像結構通常已經很強)
        for param in self.control_model_image.parameters():
            param.requires_grad = False
        
    # 2. 確保 Mask 分支是開啟訓練的
        for param in self.control_model.parameters():
            param.requires_grad = True
        
    # 3. 確保融合參數 alpha 是開啟訓練的
        self.control_alpha.requires_grad = True
    
    # --- 參數收集 ---
    # 只把需要訓練的參數丟進優化器
        params = list(self.control_model.parameters())
        params += [self.control_alpha] 
    
    # 如果 diffusion_model 本身沒有鎖定，也加入輸出層參數
        if not self.sd_locked:
        # 這裡也要確保這些參數的 requires_grad 是 True
            for param in self.model.diffusion_model.output_blocks.parameters():
                param.requires_grad = True
            for param in self.model.diffusion_model.out.parameters():
                param.requires_grad = True
            
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
    
    # --- 建立優化器 ---
    # 建議使用過濾後的列表，確保優化器不會去處理不需要梯度的參數
        trainable_params = [p for p in params if p.requires_grad]
        opt = torch.optim.AdamW(trainable_params, lr=lr)
    
        return opt
#cuda()（搬入顯存）與 cpu()（移出顯存）
    def low_vram_shift(self, is_diffusing):
        if is_diffusing: #擴散/去噪過程（生成影像的主階段）
            self.model = self.model.cuda() #LatentDiffusion 類別的實例 (物理引擎 (UNet)、調度器 (Scheduler/Sampler)、蘟空間數值轉換)
            self.control_model = self.control_model.cuda() #Mask-based ControlNet
            self.control_model_image = self.control_model_image.cuda() #Image-based ControlNet / Feature Extractor
            self.first_stage_model = self.first_stage_model.cpu() #Autoencoder KL / VAE (Variational Autoencoder)
            self.cond_stage_model = self.cond_stage_model.cpu() # CLIP text encoder
        else: #編碼/解碼過程
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.control_model_image = self.control_model_image.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()

    def p_losses(self, x_start, cond, t, noise=None):
        # 1. 準備條件字典 (直接使用完整的融合條件)
        # 這樣只需要呼叫一次 apply_model，大幅節省顯存
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 核心：只跑一次模型
        model_output = self.apply_model(x_noisy, t, cond)

        # 2. 設定學習目標
        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3]) #影像損失」：針對全圖像素的誤差
        
        # 4. --- CVC Focus Loss 強化邏輯 ---
        # A. 提取並處理遮罩
        mask = cond["c_concat_mask"][0].to(device=x_start.device, dtype=x_start.dtype) #將原始的遮罩數據轉換成模型可以運算的數學張量（Tensor）
        
        # 檢查並計算張量（Tensor）內部每一個像素格裡的「數值」。確保 mask 是 0~1 的二值圖
        if mask.max() > 1.0: mask = mask / 255.0
        mask = (mask > 0.5).float()

        # 調整尺寸以符合 Latent Space (如果需要)
        if mask.shape[-2:] != x_start.shape[-2:]:
            mask = torch.nn.functional.interpolate(mask, size=x_start.shape[-2:], mode="nearest")
        
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)

        # B. 製作平滑的權重圖
        # 使用高斯模糊讓導管邊緣有過渡，避免生成斷裂
        smoothed_mask = TF.gaussian_blur(mask, kernel_size=[5, 5], sigma=[1.5, 1.5])
        
        # C. 計算局部加權損失 (Focus Loss)
        # 只針對導管區域計算額外的 MSE，這會強迫模型「看重」導管的細節
        pixel_error = (model_output - target)**2
        loss_cvc_focus = (pixel_error * smoothed_mask).mean([1, 2, 3])

        # 5. 最終損失組合
        # 基礎損失 + (加權係數 * 導管專用損失)
        # 建議 cvc_weight 設為 2.0 ~ 5.0，讓導管特徵更突出
        cvc_lambda = getattr(self, "cvc_weight", 3.0)
        total_loss_simple = loss_simple + cvc_lambda * loss_cvc_focus

        # 6. Log 與 最終加權 (配合 LDM 原本的 VLB 邏輯)
        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        
        logvar_t = self.logvar[t].to(self.device)
        weighted_simple = (total_loss_simple / torch.exp(logvar_t) + logvar_t).mean()
        
        # VLB 通常用於微調結構，這裡使用基礎 loss_simple 即可
        weighted_vlb = (self.lvlb_weights[t] * loss_simple).mean()
        
        total_final_loss = self.l_simple_weight * weighted_simple + self.original_elbo_weight * weighted_vlb

        # 7. 更新記錄
        loss_dict.update({
            f'{prefix}/loss': total_final_loss,
            f'{prefix}/loss_simple': loss_simple.mean(),
            f'{prefix}/loss_cvc_focus': loss_cvc_focus.mean(),
            f'{prefix}/alpha': torch.sigmoid(self.control_alpha).item() # 順便監控融合權重
        })

        return total_final_loss, loss_dict