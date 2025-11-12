from typing import List
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.nvtx_utils import nvtx_range


from peft import LoraConfig, get_peft_model
from model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from model.llava.constants import IGNORE_INDEX
from datasets.utils_llcb import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

class LisaMetaModel:
    def __init__(self, config, **kwargs):
        super(LisaMetaModel, self).__init__(config)

        self.config = config
        self.config.out_dim = kwargs["out_dim"]
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # Projection layer
        in_dim = config.hidden_size
        out_dim = self.config.out_dim
        text_fc = [nn.Linear(in_dim, out_dim)]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()

        # Action prediction
        self.pred_act_mlps = nn.Linear(in_dim, in_dim//2)
        self.pred_pos_act = nn.Linear(in_dim//2, 3) # arm action
        self.pred_rot_act = nn.Linear(in_dim//2, 6) # arm action
        self.pred_gripper_act = nn.Linear(in_dim//2, 1) # gripper action (binary)

        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
        self.seg_token_idx = 32003

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def _log_vlm_outputs(self, tokenizer, prediction_ids, mask, prefix="VLM"):
        if tokenizer is None or prediction_ids is None or mask is None:
            return

        with torch.no_grad():
            tokens_cpu = prediction_ids.detach().cpu()
            mask_cpu = mask.detach().bool().cpu()

        if mask_cpu.shape != tokens_cpu.shape:
            # Align the mask to the token tensor by truncating or padding with
            # ``False`` values.  Some training setups only produce supervision
            # for the response portion of the sequence, leading to label
            # tensors that are shorter than the model's decoded output.  The
            # logging path should remain robust in those cases instead of
            # raising a shape mismatch error.
            if mask_cpu.shape[-1] > tokens_cpu.shape[-1]:
                mask_cpu = mask_cpu[..., : tokens_cpu.shape[-1]]
            else:
                pad_width = tokens_cpu.shape[-1] - mask_cpu.shape[-1]
                if pad_width:
                    pad_shape = list(mask_cpu.shape[:-1]) + [pad_width]
                    pad_tensor = torch.zeros(pad_shape, dtype=torch.bool)
                    mask_cpu = torch.cat([mask_cpu, pad_tensor], dim=-1)

        for idx in range(tokens_cpu.size(0)):
            active_mask = mask_cpu[idx]
            if not active_mask.any():
                decoded_text = ""
                token_count = 0
            else:
                decoded_text = tokenizer.decode(
                    tokens_cpu[idx][active_mask].tolist(),
                    skip_special_tokens=True,
                )
                token_count = int(active_mask.sum().item())

            print(f"[{prefix}][Sample {idx}] Output: {decoded_text}")
            print(f"[{prefix}][Sample {idx}] Token count: {token_count}")

    def _log_vlm_generated_outputs(
        self,
        tokenizer,
        input_ids,
        attention_masks,
        images,
        prefix="VLM",
    ):
        if tokenizer is None or input_ids is None or attention_masks is None:
            return

        max_new_tokens = getattr(self.config, "logging_max_new_tokens", 128)

        try:
            generated = self.generate(
                input_ids=input_ids,
                attention_mask=attention_masks.long() if attention_masks is not None else None,
                images=images,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.config.pad_token_id,
                eos_token_id=self.config.eos_token_id,
            )
        except Exception as exc:  # pragma: no cover - logging path only
            print(f"[{prefix}] Failed to generate VLM response: {exc}")
            return

        if generated.size(-1) <= input_ids.size(-1):
            print(f"[{prefix}] No generated tokens to log.")
            return

        response_tokens = generated[:, input_ids.size(-1) :]
        response_mask = response_tokens.ne(tokenizer.pad_token_id)

        if tokenizer.eos_token_id is not None:
            eos_mask = response_tokens.eq(tokenizer.eos_token_id)
            for row in range(response_tokens.size(0)):
                eos_positions = torch.nonzero(eos_mask[row], as_tuple=False)
                if eos_positions.numel() > 0:
                    first_eos = eos_positions[0].item()
                    response_mask[row, first_eos + 1 :] = False

        self._log_vlm_outputs(
            tokenizer,
            response_tokens,
            response_mask,
            prefix=prefix,
        )

    def model_forward(self,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        tokenizer,
        **kwargs,
    ):
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        
        seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 256)).bool().cuda(), seg_token_mask], dim=1,) #[bs, 255+sequence_length] 255+82=337
        
        with nvtx_range("VLM:prefilling"):
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                output_hidden_states=True,
            )

        if tokenizer is not None:
            predictions = output.logits.argmax(dim=-1)
            if labels is not None:
                mask = labels.ne(IGNORE_INDEX)
                self._log_vlm_outputs(tokenizer, predictions, mask)
            else:
                self._log_vlm_generated_outputs(
                    tokenizer,
                    input_ids,
                    attention_masks,
                    images_clip,
                    prefix="VLM",
                )

        with nvtx_range("VLM:decoding"):
            output_hidden_states = output.hidden_states

            hidden_states = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1].float()))
            action_latents = self.model.pred_act_mlps(output_hidden_states[-1][seg_token_mask].float())
            pos_pred = self.model.pred_pos_act(action_latents)
            rot_pred = self.model.pred_rot_act(action_latents)
            gripper_pred = self.model.pred_gripper_act(action_latents)
            act_pred = torch.cat([pos_pred,rot_pred,gripper_pred],dim=-1)

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

            pred_embeddings = last_hidden_state[seg_token_mask]
        ce_loss = 0
        
        return pred_embeddings, ce_loss, act_pred
    
    def evaluate(
        self,
        images_clip,
        input_ids,
        attention_masks=None,
        tokenizer=None,
        labels=None,
        **kwargs,
    ):
        with torch.no_grad():
            if attention_masks is None:
                pad_token_id = tokenizer.pad_token_id if tokenizer is not None else self.config.pad_token_id
                if pad_token_id is not None:
                    attention_masks = input_ids.ne(pad_token_id)
                else:
                    attention_masks = torch.ones_like(input_ids, dtype=torch.bool)

            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 256)).bool().cuda(), seg_token_mask], dim=1,) #[bs, 255+sequence_length] 255+82=337

            with nvtx_range("VLM:prefilling"):
                output = super().forward(
                    images=images_clip,
                    attention_mask=attention_masks,
                    input_ids=input_ids,
                    output_hidden_states=True)

            if tokenizer is not None:
                predictions = output.logits.argmax(dim=-1)
                if labels is not None:
                    mask = labels.ne(IGNORE_INDEX)
                    self._log_vlm_outputs(
                        tokenizer,
                        predictions,
                        mask,
                        prefix="VLM-Eval",
                    )
                else:
                    self._log_vlm_generated_outputs(
                        tokenizer,
                        input_ids,
                        attention_masks,
                        images_clip,
                        prefix="VLM-Eval",
                    )

            with nvtx_range("VLM:decoding"):
                output_hidden_states = output.hidden_states
                hidden_states = []

                assert len(self.model.text_hidden_fcs) == 1
                hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
                last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                pred_embeddings = last_hidden_state[seg_token_mask]

        return None, pred_embeddings


