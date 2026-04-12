import os

from CP.advlora import iter_advlora_layers
from CP.med import BertConfig, BertModel
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from CP.blip import create_vit, init_tokenizer, load_checkpoint


class BLIP_Retrieval(nn.Module):
    def __init__(
        self,
        config=None,
        med_config='configs/med_config.json',
        image_size=384,
        vit='base',
        vit_grad_ckpt=False,
        vit_ckpt_layer=0,
        embed_dim=256,
        queue_size=57600,
        momentum=0.995,
        negative_all_rank=False,
        device=None,
    ):
        super().__init__()

        self.config = config
        self.image_size = image_size
        self.lora_rank = config['lora_rank']
        self.advlora_initialized = False

        self.visual_encoder, vision_width = create_vit(
            vit,
            image_size,
            vit_grad_ckpt,
            vit_ckpt_layer,
            config=config,
            device=device,
            R=self.lora_rank,
        )

        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width

        self.text_encoder = BertModel(
            config=med_config,
            add_pooling_layer=False,
            device=device,
            R=self.lora_rank,
        )
        text_width = self.text_encoder.config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.itm_head = nn.Linear(text_width, 2)

        self.visual_encoder_m, vision_width = create_vit(
            vit,
            image_size,
            config=config,
            device=device,
            R=self.lora_rank,
        )
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(
            config=med_config,
            add_pooling_layer=False,
            device=device,
            R=self.lora_rank,
        )
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.model_pairs = [
            [self.visual_encoder, self.visual_encoder_m],
            [self.vision_proj, self.vision_proj_m],
            [self.text_encoder, self.text_encoder_m],
            [self.text_proj, self.text_proj_m],
        ]
        self.copy_params()

        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.queue_size = queue_size
        self.momentum = momentum

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.negative_all_rank = negative_all_rank

    def initialize_advlora(self):
        if self.advlora_initialized:
            return

        alpha_init = self.config.get('adaptive_init', 1e-3)
        kmeans_iters = self.config.get('cluster_iters', 10)

        for encoder in (
            self.visual_encoder,
            self.visual_encoder_m,
            self.text_encoder,
            self.text_encoder_m,
        ):
            encoder.enable_advlora(
                rank=self.lora_rank,
                alpha_init=alpha_init,
                kmeans_iters=kmeans_iters,
            )

        self.copy_params()
        self.advlora_initialized = True

    def configure_trainable_parameters(self):
        total_param = 0
        for name, parameter in self.named_parameters():
            trainable = (
                '_m' not in name
                and (
                    'lora_A' in name
                    or 'lora_B' in name
                    or 'lora_alpha' in name
                    or name.startswith('vision_proj')
                    or name.startswith('text_proj')
                    or name.startswith('itm_head')
                )
            )
            parameter.requires_grad = trainable
            if trainable:
                total_param += parameter.numel()
        return total_param

    def alignment_loss(self):
        total_loss = self.temp.new_tensor(0.0)
        for module in iter_advlora_layers(self.visual_encoder):
            total_loss = total_loss + module.align_loss()
        for module in iter_advlora_layers(self.text_encoder):
            total_loss = total_loss + module.align_loss()
        return total_loss

    def imgfea2videofea(self, tensor):
        return tensor.mean(dim=1)

    def forward(self, image, caption, alpha, idx, update_queue=True):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        if self.config['modality'] == 'image':
            image = image.unsqueeze(1)

        original_shape = image.shape
        image = image.reshape(
            image.size(0) * image.size(1),
            3,
            self.image_size,
            self.image_size,
        )

        image_embeds = self.visual_encoder(image)
        image_embeds = image_embeds.reshape(
            original_shape[0],
            original_shape[1],
            image_embeds.shape[-2],
            image_embeds.shape[-1],
        )

        image_feat = self.imgfea2videofea(image_embeds[:, :, 0, :])
        frame_feat = F.normalize(self.vision_proj(image_embeds[:, :, 0, :]), dim=-1)
        image_feat = F.normalize(self.vision_proj(image_feat), dim=-1)
        image_atts = torch.ones(
            (image_embeds.size(0), image_embeds.size(2)),
            dtype=torch.long,
            device=image.device,
        )

        text = self.tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=35,
            return_tensors="pt",
        ).to(image.device)

        text_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode='text',
        )
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)

        idx = idx.view(-1, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
        pos_idx = torch.eq(idx, idx_all).float()
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        with torch.no_grad():
            if update_queue:
                self._momentum_update()

            image_embeds_m = self.visual_encoder_m(image)
            image_embeds_m = image_embeds_m.reshape(
                original_shape[0],
                original_shape[1],
                image_embeds_m.shape[-2],
                image_embeds_m.shape[-1],
            )

            image_feat_m = self.imgfea2videofea(image_embeds_m[:, :, 0, :])
            image_feat_m = F.normalize(self.vision_proj_m(image_feat_m), dim=-1)
            image_feat_m_all = torch.cat(
                [image_feat_m.t(), self.image_queue.clone().detach()],
                dim=1,
            )

            text_output_m = self.text_encoder_m(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode='text',
            )
            text_feat_m = F.normalize(
                self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]),
                dim=-1,
            )
            text_feat_m_all = torch.cat(
                [text_feat_m.t(), self.text_queue.clone().detach()],
                dim=1,
            )

            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_m_all / self.temp
        sim_t2i = text_feat @ image_feat_m_all / self.temp

        sim_t2f = torch.einsum('bc,btc->bt', text_feat, frame_feat)

        loss_i2t = -torch.sum(
            F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets,
            dim=1,
        ).mean()
        loss_t2i = -torch.sum(
            F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets,
            dim=1,
        ).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2

        idxs = concat_all_gather(idx)
        if update_queue:
            self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)

        image_embeds = image_embeds.reshape(image_embeds.shape[0], -1, image_embeds.shape[-1])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        bs = image_embeds.size(0)

        output_pos = self.text_encoder(
            encoder_input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            frame_aware_attention_weight=sim_t2f if self.config['frame_aware_attention'] else None,
        )

        all_context = text_output.last_hidden_state[:, 0, :].mean(dim=0)
        output_pos.last_hidden_state[:, 0, :] += 0.1 * all_context

        if self.negative_all_rank:
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t())
                image_feat_world = concat_all_gather(image_feat)
                text_feat_world = concat_all_gather(text_feat)
                if self.config['frame_aware_attention']:
                    frame_feat_world = concat_all_gather(frame_feat)
                sim_i2t = image_feat @ text_feat_world.t() / self.temp
                sim_t2i = text_feat @ image_feat_world.t() / self.temp
                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)
            image_embeds_world = all_gather_with_grad(image_embeds)

            image_embeds_neg = []
            if self.config['frame_aware_attention']:
                frame_feats_neg = []
                text_feats_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds_world[neg_idx])
                if self.config['frame_aware_attention']:
                    frame_feats_neg.append(frame_feat_world[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
            if self.config['frame_aware_attention']:
                frame_feats_neg = torch.stack(frame_feats_neg, dim=0)

            input_ids_world = concat_all_gather(encoder_input_ids)
            att_mask_world = concat_all_gather(text.attention_mask)

            if self.config['frame_aware_attention']:
                text_feat_world = concat_all_gather(text_feat)
            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(input_ids_world[neg_idx])
                text_atts_neg.append(att_mask_world[neg_idx])
                if self.config['frame_aware_attention']:
                    text_feats_neg.append(text_feat_world[neg_idx])
        else:
            with torch.no_grad():
                mask = torch.eq(idx, idx.t())
                sim_i2t = image_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ image_feat.t() / self.temp

                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)

                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            image_embeds_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                image_embeds_neg.append(image_embeds[neg_idx])
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

            text_ids_neg = []
            text_atts_neg = []
            for b in range(bs):
                neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(text.attention_mask[neg_idx])

        if self.config['frame_aware_attention']:
            text_feats_neg = torch.stack(text_feats_neg, dim=0)

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        frame_aware_attention_weight = None
        if self.config['frame_aware_attention']:
            frame_feats_all = torch.cat([frame_feats_neg, frame_feat], dim=0)
            text_feats_all = torch.cat([text_feat, text_feats_neg], dim=0)
            frame_aware_attention_weight = torch.einsum('bc,btc->bt', text_feats_all, frame_feats_all)

        output_neg = self.text_encoder(
            text_ids_all,
            attention_mask=text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
            frame_aware_attention_weight=frame_aware_attention_weight,
        )

        vl_embeddings = torch.cat(
            [output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]],
            dim=0,
        )
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]
        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0

        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size
        self.ptr_queue[0] = ptr


def _pretrained_has_advlora(pretrained):
    if not pretrained or not os.path.isfile(pretrained):
        return False
    checkpoint = torch.load(pretrained, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    return any('.base.weight' in key or 'lora_A' in key for key in state_dict.keys())


def blip_retrieval(pretrained='', **kwargs):
    model = BLIP_Retrieval(**kwargs)
    if _pretrained_has_advlora(pretrained):
        model.initialize_advlora()
        model, msg = load_checkpoint(model, pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    else:
        if pretrained:
            model, msg = load_checkpoint(model, pretrained)
            print("missing keys:")
            print(msg.missing_keys)
        model.initialize_advlora()
    return model


@torch.no_grad()
def concat_all_gather(tensor):
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_with_grad(tensors):
    if not dist.is_available() or not dist.is_initialized():
        return tensors
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensors
    tensor_all = GatherLayer.apply(tensors)
    return torch.cat(tensor_all, dim=0)
