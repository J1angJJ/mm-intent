# “gesture+text+scene”作为锚点，“audio、imu”作为残差辅助

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

import baseline_real_scene as base
from project_paths import MODEL_OUTPUT_ROOT


def synchronize_device() -> None:
    """Synchronize CUDA so wall-clock measurements include queued GPU work."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def safe_divide(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


# ============================================================
# 1. Config
# ============================================================
OUTPUT_DIR = Path(
    os.getenv(
        "SMART_AR_MODEL_OUTPUT_DIR",
        str(MODEL_OUTPUT_ROOT / "improved_real_scene_anchor2_perceiver_io"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIM = int(os.getenv("IMPROVED_REAL_SCENE_A2_MODEL_DIM", str(base.MODEL_DIM)))
NUM_LATENTS = int(os.getenv("IMPROVED_REAL_SCENE_A2_NUM_LATENTS", str(base.NUM_LATENTS)))
NUM_HEADS = int(os.getenv("IMPROVED_REAL_SCENE_A2_NUM_HEADS", str(base.NUM_HEADS)))
DEPTH = int(os.getenv("IMPROVED_REAL_SCENE_A2_DEPTH", str(base.DEPTH)))
DROPOUT = float(os.getenv("IMPROVED_REAL_SCENE_A2_DROPOUT", str(base.DROPOUT)))

BATCH_SIZE = int(os.getenv("IMPROVED_REAL_SCENE_A2_BATCH_SIZE", str(base.BATCH_SIZE)))
EPOCHS = int(os.getenv("IMPROVED_REAL_SCENE_A2_EPOCHS", str(base.EPOCHS)))
PATIENCE = int(os.getenv("IMPROVED_REAL_SCENE_A2_PATIENCE", "4"))
LEARNING_RATE = float(os.getenv("IMPROVED_REAL_SCENE_A2_LR", "5e-4"))
WEIGHT_DECAY = float(os.getenv("IMPROVED_REAL_SCENE_A2_WEIGHT_DECAY", "3e-4"))
LABEL_SMOOTHING = float(os.getenv("IMPROVED_REAL_SCENE_A2_LABEL_SMOOTHING", "0.03"))
GRAD_CLIP_NORM = float(os.getenv("IMPROVED_REAL_SCENE_A2_GRAD_CLIP", "1.0"))

MIN_GATE = float(os.getenv("IMPROVED_REAL_SCENE_A2_MIN_GATE", "0.02"))
IMU_DROP_PROB = float(os.getenv("IMPROVED_REAL_SCENE_A2_IMU_DROP", "0.35"))
AUDIO_DROP_PROB = float(os.getenv("IMPROVED_REAL_SCENE_A2_AUDIO_DROP", "0.20"))
IMU_MAX_SCALE = float(os.getenv("IMPROVED_REAL_SCENE_A2_IMU_MAX_SCALE", "0.15"))
AUDIO_MAX_SCALE = float(os.getenv("IMPROVED_REAL_SCENE_A2_AUDIO_MAX_SCALE", "0.10"))
INTENT_AUX_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_INTENT_AUX_WEIGHT", "0.35"))
SCENE_AUX_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_SCENE_AUX_WEIGHT", "0.15"))
GESTURE_INTENT_AUX_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_GESTURE_INTENT_AUX_WEIGHT", "0.25"))
BASE_INTENT_AUX_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_BASE_INTENT_AUX_WEIGHT", "0.10"))
INTENT_REFINE_SCALE = float(os.getenv("IMPROVED_REAL_SCENE_A2_INTENT_REFINE_SCALE", "0.35"))
GESTURE_LOGIT_BLEND = float(os.getenv("IMPROVED_REAL_SCENE_A2_GESTURE_LOGIT_BLEND", "0.30"))
SELECTION_INTENT_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_SELECTION_INTENT_WEIGHT", "0.35"))
SELECTION_SCENE_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_SELECTION_SCENE_WEIGHT", "0.05"))
CONSISTENCY_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_CONSISTENCY_WEIGHT", "0.0"))
CONSISTENCY_MASK_PROB = float(os.getenv("IMPROVED_REAL_SCENE_A2_CONSISTENCY_MASK_PROB", "0.25"))
CONSISTENCY_NOISE_STD = float(os.getenv("IMPROVED_REAL_SCENE_A2_CONSISTENCY_NOISE_STD", "0.05"))
CONSISTENCY_TEMPERATURE = float(os.getenv("IMPROVED_REAL_SCENE_A2_CONSISTENCY_TEMPERATURE", "2.0"))
CONSISTENCY_INTENT_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_CONSISTENCY_INTENT_WEIGHT", "0.35"))
CONSISTENCY_SCENE_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_CONSISTENCY_SCENE_WEIGHT", "0.15"))
CONSISTENCY_MODALITIES = tuple(
    item.strip()
    for item in os.getenv("IMPROVED_REAL_SCENE_A2_CONSISTENCY_MODALITIES", "imu,audio").split(",")
    if item.strip()
)
MARGIN_LOSS_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_MARGIN_LOSS_WEIGHT", "0.0"))
MARGIN_VALUE = float(os.getenv("IMPROVED_REAL_SCENE_A2_MARGIN_VALUE", "0.35"))
MARGIN_INTENT_CONFUSION_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_MARGIN_INTENT_CONFUSION_WEIGHT", "1.0"))
MARGIN_SCENE_CONFUSION_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_MARGIN_SCENE_CONFUSION_WEIGHT", "1.0"))
MISSING_DISTILL_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_WEIGHT", "0.0"))
MISSING_DISTILL_TEMPERATURE = float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_TEMPERATURE", "2.0"))
MISSING_DISTILL_INTENT_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_INTENT_WEIGHT", "0.35"))
MISSING_DISTILL_SCENE_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_SCENE_WEIGHT", "0.15"))
MISSING_DISTILL_FORCE_MASK = os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_FORCE_MASK", "1") == "1"
MISSING_DISTILL_MODALITIES = tuple(
    item.strip()
    for item in os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_MODALITIES", "text,scene,audio,gesture,imu").split(",")
    if item.strip()
)
MISSING_DISTILL_PROBS = {
    "imu": float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_IMU_PROB", "0.10")),
    "gesture": float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_GESTURE_PROB", "0.10")),
    "audio": float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_AUDIO_PROB", "0.15")),
    "text": float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_TEXT_PROB", "0.35")),
    "scene": float(os.getenv("IMPROVED_REAL_SCENE_A2_MISSING_DISTILL_SCENE_PROB", "0.25")),
}
FOCAL_LOSS_GAMMA = float(os.getenv("IMPROVED_REAL_SCENE_A2_FOCAL_LOSS_GAMMA", "0.0"))
FOCAL_LOSS_APPLY_AUX = os.getenv("IMPROVED_REAL_SCENE_A2_FOCAL_LOSS_APPLY_AUX", "1") == "1"
FALLBACK_MAX_GATE = float(os.getenv("IMPROVED_REAL_SCENE_A2_FALLBACK_MAX_GATE", "0.0"))
FALLBACK_AUX_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_FALLBACK_AUX_WEIGHT", "0.0"))
SUPCON_LOSS_WEIGHT = float(os.getenv("IMPROVED_REAL_SCENE_A2_SUPCON_LOSS_WEIGHT", "0.0"))
SUPCON_TEMPERATURE = float(os.getenv("IMPROVED_REAL_SCENE_A2_SUPCON_TEMPERATURE", "0.1"))
SUPCON_TARGET = os.getenv("IMPROVED_REAL_SCENE_A2_SUPCON_TARGET", "joint").strip().lower()

MODALITY_KEYS = ("imu", "gesture", "audio", "text", "scene")
MODALITY_DISPLAY_NAMES = base.MODALITY_DISPLAY_NAMES
DEVICE = base.DEVICE
SKIP_TEST_EVAL = os.getenv("SMART_AR_SKIP_TEST_EVAL", "0") == "1"


# ============================================================
# 2. Dataset / DataLoader
# ============================================================
class MultimodalSceneIntentDataset(Dataset):
    def __init__(
        self,
        features: Dict[str, np.ndarray],
        joint_labels: np.ndarray,
        intent_targets: np.ndarray,
        scene_targets: np.ndarray,
    ):
        self.imu = torch.from_numpy(features["imu"].astype(np.float32))
        self.gesture = torch.from_numpy(features["gesture"].astype(np.float32))
        self.audio = torch.from_numpy(features["audio"].astype(np.float32))
        self.text = torch.from_numpy(features["text"].astype(np.float32))
        self.scene = torch.from_numpy(features["scene"].astype(np.float32))
        self.joint_labels = torch.from_numpy(joint_labels.astype(np.int64))
        self.intent_targets = torch.from_numpy(intent_targets.astype(np.int64))
        self.scene_targets = torch.from_numpy(scene_targets.astype(np.int64))

    def __len__(self) -> int:
        return len(self.joint_labels)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.imu[index],
            self.gesture[index],
            self.audio[index],
            self.text[index],
            self.scene[index],
            self.joint_labels[index],
            self.intent_targets[index],
            self.scene_targets[index],
        )


def make_loader(
    features: Dict[str, np.ndarray],
    joint_labels: np.ndarray,
    intent_targets: np.ndarray,
    scene_targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = MultimodalSceneIntentDataset(features, joint_labels, intent_targets, scene_targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# 3. Improved model
# ============================================================
class ScalarGate(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        hidden_dim = max(model_dim // 2, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(model_dim * 2),
            nn.Linear(model_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, modality_summary: torch.Tensor, anchor_summary: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([modality_summary, anchor_summary], dim=-1))
        return torch.sigmoid(logits)


class CrossReadoutBlock(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, query_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_dim * 2, query_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        x = attn_out
        x = x + self.ff(x)
        return x


class Anchor2PerceiverIO(nn.Module):
    def __init__(
        self,
        num_joint_classes: int,
        num_intent_classes: int,
        num_scene_classes: int,
        joint_class_names: List[str],
        model_dim: int = MODEL_DIM,
        num_latents: int = NUM_LATENTS,
        depth: int = DEPTH,
        num_heads: int = NUM_HEADS,
        dropout: float = DROPOUT,
        min_gate: float = MIN_GATE,
    ):
        super().__init__()
        self.min_gate = min_gate
        self.modality_drop_probs = {
            "imu": IMU_DROP_PROB,
            "audio": AUDIO_DROP_PROB,
        }
        self.support_scales = {
            "imu": IMU_MAX_SCALE,
            "audio": AUDIO_MAX_SCALE,
        }

        self.imu_proj = nn.Sequential(nn.Linear(base.IMU_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.gesture_proj = nn.Sequential(nn.Linear(base.GESTURE_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.audio_proj = nn.Sequential(nn.Linear(base.AUDIO_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.text_proj = nn.Sequential(nn.Linear(base.TEXT_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))
        self.scene_proj = nn.Sequential(nn.Linear(base.SCENE_FEAT_DIM, model_dim), nn.LayerNorm(model_dim))

        self.time_embedding = nn.Parameter(torch.randn(1, base.TARGET_TIMESTEPS, model_dim) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(len(MODALITY_KEYS), 1, model_dim) * 0.02)
        self.input_dropout = nn.Dropout(dropout)

        self.anchor_encoder = base.PerceiverEncoder(
            latent_dim=model_dim,
            num_latents=num_latents,
            depth=depth,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.output_query = nn.Parameter(torch.randn(1, 1, model_dim) * 0.02)
        self.anchor_decoder = base.CrossAttentionBlock(model_dim, model_dim, num_heads, dropout)
        self.support_readers = nn.ModuleDict(
            {key: CrossReadoutBlock(model_dim, model_dim, num_heads, dropout) for key in ("imu", "audio")}
        )
        self.gates = nn.ModuleDict(
            {key: ScalarGate(model_dim, dropout) for key in ("imu", "audio")}
        )
        self.gesture_readout = CrossReadoutBlock(model_dim, model_dim, num_heads, dropout)

        self.fusion_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_intent_classes),
        )
        self.intent_refine_gate = ScalarGate(model_dim, dropout)
        self.intent_refine_head = nn.Sequential(
            nn.LayerNorm(model_dim * 4),
            nn.Linear(model_dim * 4, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, num_intent_classes),
        )
        self.scene_classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_scene_classes),
        )
        self.gesture_intent_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_intent_classes),
        )
        self.fallback_encoder = nn.Sequential(
            nn.LayerNorm(model_dim * len(MODALITY_KEYS)),
            nn.Linear(model_dim * len(MODALITY_KEYS), model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
        )
        self.fallback_intent_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_intent_classes),
        )
        self.fallback_scene_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, num_scene_classes),
        )
        self.fallback_gate = nn.Sequential(
            nn.LayerNorm(model_dim * 3),
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1),
        )

        scene_name_to_idx = {name: idx for idx, name in enumerate([base.SCENE_ID_TO_NAME[i] for i in range(num_scene_classes)])}
        intent_name_to_idx = {name: idx for idx, name in enumerate([base.INTENT_NAMES[i] for i in range(num_intent_classes)])}
        joint_scene_index = []
        joint_intent_index = []
        for joint_name in joint_class_names:
            scene_name, intent_name = base.split_joint_label(joint_name)
            joint_scene_index.append(scene_name_to_idx[scene_name])
            joint_intent_index.append(intent_name_to_idx[intent_name])
        self.register_buffer("joint_scene_index", torch.tensor(joint_scene_index, dtype=torch.long))
        self.register_buffer("joint_intent_index", torch.tensor(joint_intent_index, dtype=torch.long))

    def _add_embeddings(self, tokens: torch.Tensor, modality_index: int) -> torch.Tensor:
        return tokens + self.time_embedding + self.modality_embedding[modality_index]

    def _add_single_token_embedding(self, token: torch.Tensor, modality_index: int) -> torch.Tensor:
        return token + self.modality_embedding[modality_index]

    def _sample_keep_masks(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        for key in ("imu", "audio"):
            drop_prob = self.modality_drop_probs[key] if self.training else 0.0
            if drop_prob <= 0.0:
                keep = torch.ones(batch_size, 1, 1, device=device)
            else:
                keep = (torch.rand(batch_size, 1, 1, device=device) > drop_prob).float()
            masks[key] = keep
        return masks

    def _gate_value(self, key: str, modality_summary: torch.Tensor, anchor_summary: torch.Tensor) -> torch.Tensor:
        gate = self.gates[key](modality_summary, anchor_summary)
        return self.min_gate + (1.0 - self.min_gate) * gate

    def forward(
        self,
        imu: torch.Tensor,
        gesture: torch.Tensor,
        audio: torch.Tensor,
        text: torch.Tensor,
        scene: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        imu_tokens = self._add_embeddings(self.imu_proj(imu), 0)
        gesture_tokens = self._add_embeddings(self.gesture_proj(gesture), 1)
        audio_tokens = self._add_embeddings(self.audio_proj(audio), 2)
        text_tokens = self._add_embeddings(self.text_proj(text), 3)
        scene_token = self._add_single_token_embedding(self.scene_proj(scene).unsqueeze(1), 4)

        keep_masks = self._sample_keep_masks(imu_tokens.shape[0], imu_tokens.device)
        imu_tokens = imu_tokens * keep_masks["imu"]
        audio_tokens = audio_tokens * keep_masks["audio"]

        anchor_tokens = torch.cat([gesture_tokens, text_tokens, scene_token], dim=1)
        anchor_tokens = self.input_dropout(anchor_tokens)
        latents = self.anchor_encoder(anchor_tokens)
        query = self.output_query.expand(anchor_tokens.shape[0], -1, -1)
        anchor_repr = self.anchor_decoder(query, latents).squeeze(1)
        gesture_support_feature = self.gesture_readout(anchor_repr.unsqueeze(1), gesture_tokens).squeeze(1)

        support_tokens = {
            "imu": imu_tokens,
            "audio": audio_tokens,
        }
        residual = torch.zeros_like(anchor_repr)
        gate_values: Dict[str, torch.Tensor] = {}
        for key, tokens in support_tokens.items():
            summary = tokens.mean(dim=1)
            support_feature = self.support_readers[key](anchor_repr.unsqueeze(1), tokens).squeeze(1)
            gate = self._gate_value(key, summary, anchor_repr)
            residual = residual + self.support_scales[key] * gate * support_feature
            gate_values[key] = gate

        fused = self.fusion_norm(anchor_repr + residual)
        base_intent_logits = self.classifier(fused)
        scene_logits = self.scene_classifier(anchor_repr)
        gesture_intent_logits = self.gesture_intent_head(gesture_support_feature)
        intent_refine_input = torch.cat(
            [
                anchor_repr,
                fused,
                gesture_support_feature,
                fused - gesture_support_feature,
            ],
            dim=-1,
        )
        intent_refine_logits = self.intent_refine_head(intent_refine_input)
        intent_refine_gate = self.intent_refine_gate(gesture_support_feature, fused)
        intent_logits = base_intent_logits + intent_refine_gate * (
            INTENT_REFINE_SCALE * intent_refine_logits
            + GESTURE_LOGIT_BLEND * gesture_intent_logits
        )
        joint_logits_main = scene_logits.index_select(1, self.joint_scene_index) + intent_logits.index_select(1, self.joint_intent_index)

        fallback_input = torch.cat(
            [
                imu_tokens.mean(dim=1),
                gesture_tokens.mean(dim=1),
                audio_tokens.mean(dim=1),
                text_tokens.mean(dim=1),
                scene_token.squeeze(1),
            ],
            dim=-1,
        )
        fallback_repr = self.fallback_encoder(fallback_input)
        fallback_intent_logits = self.fallback_intent_head(fallback_repr)
        fallback_scene_logits = self.fallback_scene_head(fallback_repr)
        fallback_joint_logits = (
            fallback_scene_logits.index_select(1, self.joint_scene_index)
            + fallback_intent_logits.index_select(1, self.joint_intent_index)
        )
        fallback_gate = FALLBACK_MAX_GATE * torch.sigmoid(
            self.fallback_gate(torch.cat([anchor_repr, fused, fallback_repr], dim=-1))
        )
        joint_logits = (1.0 - fallback_gate) * joint_logits_main + fallback_gate * fallback_joint_logits

        modality_gates = torch.cat(
            [
                gate_values["imu"],
                torch.ones_like(gate_values["imu"]),
                gate_values["audio"],
                torch.ones_like(gate_values["imu"]),
                torch.ones_like(gate_values["imu"]),
            ],
            dim=1,
        )
        return {
            "joint_logits": joint_logits,
            "main_joint_logits": joint_logits_main,
            "intent_logits": intent_logits,
            "base_intent_logits": base_intent_logits,
            "scene_logits": scene_logits,
            "gesture_intent_logits": gesture_intent_logits,
            "intent_refine_logits": intent_refine_logits,
            "intent_refine_gate": intent_refine_gate,
            "fallback_joint_logits": fallback_joint_logits,
            "fallback_intent_logits": fallback_intent_logits,
            "fallback_scene_logits": fallback_scene_logits,
            "fallback_gate": fallback_gate,
            "fused_embedding": fused,
            "anchor_embedding": anchor_repr,
            "gesture_embedding": gesture_support_feature,
            "fallback_embedding": fallback_repr,
            "modality_gates": modality_gates,
            "joint_scene_index": self.joint_scene_index,
            "joint_intent_index": self.joint_intent_index,
        }


# ============================================================
# 4. Train / evaluate
# ============================================================
class WeightedFocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.gamma = gamma
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        losses = -((1.0 - target_probs).clamp_min(1e-6) ** self.gamma) * target_log_probs
        if self.weight is not None:
            losses = losses * self.weight.to(logits.device).gather(0, targets)
        return losses.mean()


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    joint_targets: torch.Tensor,
    intent_targets: torch.Tensor,
    scene_targets: torch.Tensor,
    joint_criterion: nn.Module,
    intent_criterion: nn.Module,
    scene_criterion: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    joint_loss = joint_criterion(outputs["joint_logits"], joint_targets)
    intent_loss = intent_criterion(outputs["intent_logits"], intent_targets)
    base_intent_loss = intent_criterion(outputs["base_intent_logits"], intent_targets)
    scene_loss = scene_criterion(outputs["scene_logits"], scene_targets)
    gesture_intent_loss = intent_criterion(outputs["gesture_intent_logits"], intent_targets)
    fallback_joint_loss = joint_criterion(outputs["fallback_joint_logits"], joint_targets)
    fallback_intent_loss = intent_criterion(outputs["fallback_intent_logits"], intent_targets)
    fallback_scene_loss = scene_criterion(outputs["fallback_scene_logits"], scene_targets)
    fallback_loss = fallback_joint_loss + 0.35 * fallback_intent_loss + 0.15 * fallback_scene_loss
    total_loss = (
        joint_loss
        + INTENT_AUX_WEIGHT * intent_loss
        + BASE_INTENT_AUX_WEIGHT * base_intent_loss
        + SCENE_AUX_WEIGHT * scene_loss
        + GESTURE_INTENT_AUX_WEIGHT * gesture_intent_loss
        + FALLBACK_AUX_WEIGHT * fallback_loss
    )
    return total_loss, {
        "joint_loss": float(joint_loss.item()),
        "intent_loss": float(intent_loss.item()),
        "base_intent_loss": float(base_intent_loss.item()),
        "scene_loss": float(scene_loss.item()),
        "gesture_intent_loss": float(gesture_intent_loss.item()),
        "fallback_loss": float(fallback_loss.item()),
    }


def masked_max(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return masked.max(dim=1).values


def compute_hierarchical_margin_loss(
    outputs: Dict[str, torch.Tensor],
    joint_targets: torch.Tensor,
    intent_targets: torch.Tensor,
    scene_targets: torch.Tensor,
) -> torch.Tensor:
    joint_logits = outputs["joint_logits"]
    true_logits = joint_logits.gather(1, joint_targets.unsqueeze(1)).squeeze(1)
    joint_intent_index = outputs["joint_intent_index"].to(joint_logits.device)
    joint_scene_index = outputs["joint_scene_index"].to(joint_logits.device)

    same_intent_other_scene = (
        joint_intent_index.unsqueeze(0) == intent_targets.unsqueeze(1)
    ) & (
        joint_scene_index.unsqueeze(0) != scene_targets.unsqueeze(1)
    )
    same_scene_other_intent = (
        joint_scene_index.unsqueeze(0) == scene_targets.unsqueeze(1)
    ) & (
        joint_intent_index.unsqueeze(0) != intent_targets.unsqueeze(1)
    )

    scene_confusion_neg = masked_max(joint_logits, same_intent_other_scene)
    intent_confusion_neg = masked_max(joint_logits, same_scene_other_intent)
    scene_margin = F.relu(MARGIN_VALUE + scene_confusion_neg - true_logits)
    intent_margin = F.relu(MARGIN_VALUE + intent_confusion_neg - true_logits)
    return (
        MARGIN_SCENE_CONFUSION_WEIGHT * scene_margin.mean()
        + MARGIN_INTENT_CONFUSION_WEIGHT * intent_margin.mean()
    )


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embeddings.size(0) <= 1:
        return embeddings.new_tensor(0.0)
    features = F.normalize(embeddings, dim=1)
    logits = torch.matmul(features, features.T) / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask
    valid_anchor = positive_mask.any(dim=1)
    if not bool(valid_anchor.any().item()):
        return embeddings.new_tensor(0.0)
    exp_logits = torch.exp(logits) * (~self_mask).to(logits.dtype)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (positive_mask.to(logits.dtype) * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[valid_anchor].mean()


def perturb_batch_for_consistency(
    batch_imu: torch.Tensor,
    batch_gesture: torch.Tensor,
    batch_audio: torch.Tensor,
    batch_text: torch.Tensor,
    batch_scene: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    perturbed = []
    for key, tensor in zip(MODALITY_KEYS, (batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)):
        if key not in CONSISTENCY_MODALITIES:
            perturbed.append(tensor)
            continue
        noisy = tensor
        if CONSISTENCY_NOISE_STD > 0.0:
            reduce_dims = tuple(range(1, tensor.ndim))
            scale = tensor.detach().std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
            noisy = noisy + torch.randn_like(tensor) * scale * CONSISTENCY_NOISE_STD
        if CONSISTENCY_MASK_PROB > 0.0:
            mask_shape = (tensor.size(0),) + (1,) * (tensor.ndim - 1)
            keep = (torch.rand(mask_shape, device=tensor.device) > CONSISTENCY_MASK_PROB).to(tensor.dtype)
            noisy = noisy * keep
        perturbed.append(noisy)
    return tuple(perturbed)  # type: ignore[return-value]


def distillation_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    temperature = max(CONSISTENCY_TEMPERATURE, 1e-6)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def distillation_kl_with_temperature(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temperature = max(temperature, 1e-6)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def compute_consistency_loss(
    clean_outputs: Dict[str, torch.Tensor],
    perturbed_outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    joint_consistency = distillation_kl(
        perturbed_outputs["joint_logits"],
        clean_outputs["joint_logits"],
    )
    intent_consistency = distillation_kl(
        perturbed_outputs["intent_logits"],
        clean_outputs["intent_logits"],
    )
    scene_consistency = distillation_kl(
        perturbed_outputs["scene_logits"],
        clean_outputs["scene_logits"],
    )
    return (
        joint_consistency
        + CONSISTENCY_INTENT_WEIGHT * intent_consistency
        + CONSISTENCY_SCENE_WEIGHT * scene_consistency
    )


def mask_batch_for_missing_distillation(
    batch_imu: torch.Tensor,
    batch_gesture: torch.Tensor,
    batch_audio: torch.Tensor,
    batch_text: torch.Tensor,
    batch_scene: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = {
        "imu": batch_imu,
        "gesture": batch_gesture,
        "audio": batch_audio,
        "text": batch_text,
        "scene": batch_scene,
    }
    masked_tensors: Dict[str, torch.Tensor] = {}
    masked_any = torch.zeros(batch_imu.size(0), dtype=torch.bool, device=batch_imu.device)

    for key in MODALITY_KEYS:
        tensor = tensors[key]
        if key not in MISSING_DISTILL_MODALITIES:
            masked_tensors[key] = tensor
            continue
        probability = float(np.clip(MISSING_DISTILL_PROBS.get(key, 0.0), 0.0, 1.0))
        mask_shape = (tensor.size(0),) + (1,) * (tensor.ndim - 1)
        drop = torch.rand(mask_shape, device=tensor.device) < probability
        masked_any |= drop.flatten()
        masked_tensors[key] = tensor * (~drop).to(tensor.dtype)

    if MISSING_DISTILL_FORCE_MASK and not bool(masked_any.all().item()):
        candidates = [key for key in MODALITY_KEYS if key in MISSING_DISTILL_MODALITIES]
        if candidates:
            unmasked_indices = torch.nonzero(~masked_any, as_tuple=False).flatten()
            for sample_index in unmasked_indices.tolist():
                key = candidates[int(torch.randint(len(candidates), (1,), device=batch_imu.device).item())]
                tensor = masked_tensors[key].clone()
                tensor[sample_index] = 0
                masked_tensors[key] = tensor

    return (
        masked_tensors["imu"],
        masked_tensors["gesture"],
        masked_tensors["audio"],
        masked_tensors["text"],
        masked_tensors["scene"],
    )


def compute_missing_distillation_loss(
    teacher_outputs: Dict[str, torch.Tensor],
    student_outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    joint_loss = distillation_kl_with_temperature(
        student_outputs["joint_logits"],
        teacher_outputs["joint_logits"],
        MISSING_DISTILL_TEMPERATURE,
    )
    intent_loss = distillation_kl_with_temperature(
        student_outputs["intent_logits"],
        teacher_outputs["intent_logits"],
        MISSING_DISTILL_TEMPERATURE,
    )
    scene_loss = distillation_kl_with_temperature(
        student_outputs["scene_logits"],
        teacher_outputs["scene_logits"],
        MISSING_DISTILL_TEMPERATURE,
    )
    return (
        joint_loss
        + MISSING_DISTILL_INTENT_WEIGHT * intent_loss
        + MISSING_DISTILL_SCENE_WEIGHT * scene_loss
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    joint_criterion: nn.Module,
    intent_criterion: nn.Module,
    scene_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float, Dict[str, float]]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_meter = {
        "joint_loss": 0.0,
        "intent_loss": 0.0,
        "base_intent_loss": 0.0,
        "scene_loss": 0.0,
        "gesture_intent_loss": 0.0,
        "fallback_loss": 0.0,
        "consistency_loss": 0.0,
        "margin_loss": 0.0,
        "missing_distill_loss": 0.0,
        "supcon_loss": 0.0,
    }

    for batch in loader:
        batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_joint_y, batch_intent_y, batch_scene_y = batch
        batch_imu = batch_imu.to(DEVICE)
        batch_gesture = batch_gesture.to(DEVICE)
        batch_audio = batch_audio.to(DEVICE)
        batch_text = batch_text.to(DEVICE)
        batch_scene = batch_scene.to(DEVICE)
        batch_joint_y = batch_joint_y.to(DEVICE)
        batch_intent_y = batch_intent_y.to(DEVICE)
        batch_scene_y = batch_scene_y.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss, loss_parts = compute_loss(
            outputs,
            batch_joint_y,
            batch_intent_y,
            batch_scene_y,
            joint_criterion,
            intent_criterion,
            scene_criterion,
        )
        if CONSISTENCY_WEIGHT > 0.0:
            perturbed_inputs = perturb_batch_for_consistency(
                batch_imu,
                batch_gesture,
                batch_audio,
                batch_text,
                batch_scene,
            )
            perturbed_outputs = model(*perturbed_inputs)
            consistency_loss = compute_consistency_loss(outputs, perturbed_outputs)
            loss = loss + CONSISTENCY_WEIGHT * consistency_loss
            loss_parts["consistency_loss"] = float(consistency_loss.item())
        else:
            loss_parts["consistency_loss"] = 0.0
        if MARGIN_LOSS_WEIGHT > 0.0:
            margin_loss = compute_hierarchical_margin_loss(
                outputs,
                batch_joint_y,
                batch_intent_y,
                batch_scene_y,
            )
            loss = loss + MARGIN_LOSS_WEIGHT * margin_loss
            loss_parts["margin_loss"] = float(margin_loss.item())
        else:
            loss_parts["margin_loss"] = 0.0
        if MISSING_DISTILL_WEIGHT > 0.0:
            masked_inputs = mask_batch_for_missing_distillation(
                batch_imu,
                batch_gesture,
                batch_audio,
                batch_text,
                batch_scene,
            )
            student_outputs = model(*masked_inputs)
            missing_distill_loss = compute_missing_distillation_loss(outputs, student_outputs)
            loss = loss + MISSING_DISTILL_WEIGHT * missing_distill_loss
            loss_parts["missing_distill_loss"] = float(missing_distill_loss.item())
        else:
            loss_parts["missing_distill_loss"] = 0.0
        if SUPCON_LOSS_WEIGHT > 0.0:
            if SUPCON_TARGET == "intent":
                supcon_labels = batch_intent_y
            elif SUPCON_TARGET == "scene":
                supcon_labels = batch_scene_y
            else:
                supcon_labels = batch_joint_y
            supcon_loss = supervised_contrastive_loss(
                outputs["fused_embedding"],
                supcon_labels,
                SUPCON_TEMPERATURE,
            )
            loss = loss + SUPCON_LOSS_WEIGHT * supcon_loss
            loss_parts["supcon_loss"] = float(supcon_loss.item())
        else:
            loss_parts["supcon_loss"] = 0.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        batch_size = batch_imu.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs["joint_logits"].argmax(dim=1)
        total_correct += (preds == batch_joint_y).sum().item()
        total_samples += batch_size
        for key in loss_meter:
            loss_meter[key] += loss_parts[key] * batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    avg_parts = {key: value / max(total_samples, 1) for key, value in loss_meter.items()}
    return avg_loss, avg_acc, avg_parts


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    joint_criterion: nn.Module,
    intent_criterion: nn.Module,
    scene_criterion: nn.Module,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_meter = {
        "joint_loss": 0.0,
        "intent_loss": 0.0,
        "base_intent_loss": 0.0,
        "scene_loss": 0.0,
        "gesture_intent_loss": 0.0,
        "fallback_loss": 0.0,
    }

    joint_true: List[np.ndarray] = []
    joint_pred: List[np.ndarray] = []
    intent_true: List[np.ndarray] = []
    intent_pred: List[np.ndarray] = []
    scene_true: List[np.ndarray] = []
    scene_pred: List[np.ndarray] = []
    all_gates: List[np.ndarray] = []

    for batch in loader:
        batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_joint_y, batch_intent_y, batch_scene_y = batch
        batch_imu = batch_imu.to(DEVICE)
        batch_gesture = batch_gesture.to(DEVICE)
        batch_audio = batch_audio.to(DEVICE)
        batch_text = batch_text.to(DEVICE)
        batch_scene = batch_scene.to(DEVICE)
        batch_joint_y = batch_joint_y.to(DEVICE)
        batch_intent_y = batch_intent_y.to(DEVICE)
        batch_scene_y = batch_scene_y.to(DEVICE)

        outputs = model(batch_imu, batch_gesture, batch_audio, batch_text, batch_scene)
        loss, loss_parts = compute_loss(
            outputs,
            batch_joint_y,
            batch_intent_y,
            batch_scene_y,
            joint_criterion,
            intent_criterion,
            scene_criterion,
        )

        batch_size = batch_imu.size(0)
        total_loss += loss.item() * batch_size
        joint_preds = outputs["joint_logits"].argmax(dim=1)

        total_correct += (joint_preds == batch_joint_y).sum().item()
        total_samples += batch_size
        for key in loss_meter:
            loss_meter[key] += loss_parts[key] * batch_size

        joint_true.append(batch_joint_y.cpu().numpy())
        joint_pred.append(joint_preds.cpu().numpy())
        intent_true.append(batch_intent_y.cpu().numpy())
        intent_pred.append(outputs["intent_logits"].argmax(dim=1).cpu().numpy())
        scene_true.append(batch_scene_y.cpu().numpy())
        scene_pred.append(outputs["scene_logits"].argmax(dim=1).cpu().numpy())
        all_gates.append(outputs["modality_gates"].cpu().numpy())

    avg_parts = {key: value / max(total_samples, 1) for key, value in loss_meter.items()}
    avg_gates = np.concatenate(all_gates, axis=0).mean(axis=0).tolist() if all_gates else [0.0] * len(MODALITY_KEYS)
    joint_true_arr = np.concatenate(joint_true) if joint_true else np.array([], dtype=np.int64)
    joint_pred_arr = np.concatenate(joint_pred) if joint_pred else np.array([], dtype=np.int64)
    intent_true_arr = np.concatenate(intent_true) if intent_true else np.array([], dtype=np.int64)
    intent_pred_arr = np.concatenate(intent_pred) if intent_pred else np.array([], dtype=np.int64)
    scene_true_arr = np.concatenate(scene_true) if scene_true else np.array([], dtype=np.int64)
    scene_pred_arr = np.concatenate(scene_pred) if scene_pred else np.array([], dtype=np.int64)
    return {
        "loss": total_loss / max(total_samples, 1),
        "joint_acc": total_correct / max(total_samples, 1),
        "intent_acc": float(np.mean(intent_true_arr == intent_pred_arr)) if len(intent_true_arr) else 0.0,
        "scene_acc": float(np.mean(scene_true_arr == scene_pred_arr)) if len(scene_true_arr) else 0.0,
        "joint_true": joint_true_arr,
        "joint_pred": joint_pred_arr,
        "intent_true": intent_true_arr,
        "intent_pred": intent_pred_arr,
        "scene_true": scene_true_arr,
        "scene_pred": scene_pred_arr,
        "avg_modality_gates": avg_gates,
        **avg_parts,
    }


def evaluate_feature_subset(
    model: nn.Module,
    features: Dict[str, np.ndarray],
    joint_labels: np.ndarray,
    intent_targets: np.ndarray,
    scene_targets: np.ndarray,
    joint_criterion: nn.Module,
    intent_criterion: nn.Module,
    scene_criterion: nn.Module,
) -> Dict[str, float]:
    loader = make_loader(
        features,
        joint_labels,
        intent_targets,
        scene_targets,
        BATCH_SIZE,
        shuffle=False,
    )
    metrics = evaluate(
        model,
        loader,
        joint_criterion,
        intent_criterion,
        scene_criterion,
    )
    return {
        "loss": float(metrics["loss"]),
        "joint_acc": float(metrics["joint_acc"]),
        "intent_acc": float(metrics["intent_acc"]),
        "scene_acc": float(metrics["scene_acc"]),
    }


def evaluate_modality_subsets(
    model: nn.Module,
    features: Dict[str, np.ndarray],
    joint_labels: np.ndarray,
    intent_targets: np.ndarray,
    scene_targets: np.ndarray,
    joint_criterion: nn.Module,
    intent_criterion: nn.Module,
    scene_criterion: nn.Module,
) -> Dict[str, Dict[str, object]]:
    subset_metrics: Dict[str, Dict[str, object]] = {}
    total_subsets = 2 ** len(MODALITY_KEYS)
    subset_index = 0

    for subset_size in range(len(MODALITY_KEYS) + 1):
        for subset in base.combinations(MODALITY_KEYS, subset_size):
            subset_index += 1
            subset_name = base.subset_to_name(subset)
            print(f"[contribution] evaluate subset {subset_index:02d}/{total_subsets:02d}: {subset_name}")
            masked_features = base.mask_features_for_modalities(features, subset)
            metrics = evaluate_feature_subset(
                model,
                masked_features,
                joint_labels,
                intent_targets,
                scene_targets,
                joint_criterion,
                intent_criterion,
                scene_criterion,
            )
            subset_metrics[subset_name] = {
                "active_modalities": list(subset),
                **metrics,
            }

    return subset_metrics


# ============================================================
# 5. Main
# ============================================================
def main() -> None:
    base.set_seed(base.RANDOM_SEED)
    scene_cache = base.RealSceneFeatureCache(base.SCENE_CACHE_DIR)

    print(f"[device] {DEVICE}")
    print(
        f"[config] epochs={EPOCHS}, batch_size={BATCH_SIZE}, patience={PATIENCE}, "
        f"model_dim={MODEL_DIM}, num_latents={NUM_LATENTS}, depth={DEPTH}, heads={NUM_HEADS}"
    )
    print(
        f"[fusion] anchor=gesture+text+scene residual=imu+audio "
        f"drop_probs={{imu:{IMU_DROP_PROB}, audio:{AUDIO_DROP_PROB}}} "
        f"support_scales={{imu:{IMU_MAX_SCALE}, audio:{AUDIO_MAX_SCALE}}} "
        f"intent_aux={INTENT_AUX_WEIGHT} base_intent_aux={BASE_INTENT_AUX_WEIGHT} "
        f"scene_aux={SCENE_AUX_WEIGHT} gesture_intent_aux={GESTURE_INTENT_AUX_WEIGHT} "
        f"intent_refine_scale={INTENT_REFINE_SCALE} gesture_logit_blend={GESTURE_LOGIT_BLEND}"
    )
    print(
        f"[consistency] weight={CONSISTENCY_WEIGHT} mask_prob={CONSISTENCY_MASK_PROB} "
        f"noise_std={CONSISTENCY_NOISE_STD} temperature={CONSISTENCY_TEMPERATURE} "
        f"intent_weight={CONSISTENCY_INTENT_WEIGHT} scene_weight={CONSISTENCY_SCENE_WEIGHT} "
        f"modalities={','.join(CONSISTENCY_MODALITIES)}"
    )
    print(
        f"[hier_margin] weight={MARGIN_LOSS_WEIGHT} margin={MARGIN_VALUE} "
        f"intent_confusion_weight={MARGIN_INTENT_CONFUSION_WEIGHT} "
        f"scene_confusion_weight={MARGIN_SCENE_CONFUSION_WEIGHT}"
    )
    print(
        f"[missing_distill] weight={MISSING_DISTILL_WEIGHT} temperature={MISSING_DISTILL_TEMPERATURE} "
        f"intent_weight={MISSING_DISTILL_INTENT_WEIGHT} scene_weight={MISSING_DISTILL_SCENE_WEIGHT} "
        f"force_mask={int(MISSING_DISTILL_FORCE_MASK)} modalities={','.join(MISSING_DISTILL_MODALITIES)} "
        f"probs={MISSING_DISTILL_PROBS}"
    )
    print(
        f"[focal] gamma={FOCAL_LOSS_GAMMA} apply_aux={int(FOCAL_LOSS_APPLY_AUX)}"
    )
    print(
        f"[fallback] max_gate={FALLBACK_MAX_GATE} aux_weight={FALLBACK_AUX_WEIGHT}"
    )
    print(
        f"[supcon] weight={SUPCON_LOSS_WEIGHT} temperature={SUPCON_TEMPERATURE} target={SUPCON_TARGET}"
    )

    print("[step] load train split with real scene")
    train_raw_features, train_raw_labels, train_raw_scene_targets, train_scene_selection = base.load_multimodal_data(
        base.TRAIN_VIDEO_NAMES,
        scene_cache,
    )

    train_joint_labels_raw = base.build_joint_labels(train_raw_labels, train_raw_scene_targets)
    test_raw_features = None
    test_raw_labels = None
    test_raw_scene_targets = None
    test_scene_selection: Dict[str, Dict[str, object]] = {}
    if SKIP_TEST_EVAL:
        test_joint_labels_raw = np.array([], dtype=object)
        print("[step] skip test split for train-only timing")
    else:
        print("[step] load test split with real scene")
        test_raw_features, test_raw_labels, test_raw_scene_targets, test_scene_selection = base.load_multimodal_data(
            base.TEST_VIDEO_NAMES,
            scene_cache,
        )
        test_joint_labels_raw = base.build_joint_labels(test_raw_labels, test_raw_scene_targets)

    (
        train_features_raw,
        val_features_raw,
        y_train_raw,
        y_val_raw,
        y_train_scene_raw,
        y_val_scene_raw,
        y_train_joint_raw,
        y_val_joint_raw,
    ) = base.split_train_val(
        train_raw_features,
        train_raw_labels,
        train_raw_scene_targets,
        train_joint_labels_raw,
    )

    print("[split]")
    print(f"  train {len(y_train_joint_raw)} -> {base.summarize_joint_labels(y_train_joint_raw)}")
    print(f"  val   {len(y_val_joint_raw)} -> {base.summarize_joint_labels(y_val_joint_raw)}")
    if SKIP_TEST_EVAL:
        print("  test  skipped")
    else:
        print(f"  test  {len(test_joint_labels_raw)} -> {base.summarize_joint_labels(test_joint_labels_raw)}")

    scalers = base.fit_scalers(train_features_raw)
    train_features_scaled = base.apply_scalers(train_features_raw, scalers)
    val_features_scaled = base.apply_scalers(val_features_raw, scalers)
    test_features_scaled = base.apply_scalers(test_raw_features, scalers) if not SKIP_TEST_EVAL else None

    y_train_joint, label_encoder = base.encode_labels(y_train_joint_raw)
    y_val_joint = label_encoder.transform(y_val_joint_raw)
    joint_class_names = label_encoder.classes_.tolist()
    intent_class_names = [base.INTENT_NAMES[index] for index in sorted(base.INTENT_NAMES)]
    scene_class_names = [base.SCENE_ID_TO_NAME[index] for index in range(len(base.SCENE_ID_TO_NAME))]

    train_loader = make_loader(train_features_scaled, y_train_joint, y_train_raw, y_train_scene_raw, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_features_scaled, y_val_joint, y_val_raw, y_val_scene_raw, BATCH_SIZE, shuffle=False)
    if SKIP_TEST_EVAL:
        y_test_joint = None
        test_loader = None
    else:
        y_test_joint = label_encoder.transform(test_joint_labels_raw)
        test_loader = make_loader(test_features_scaled, y_test_joint, test_raw_labels, test_raw_scene_targets, BATCH_SIZE, shuffle=False)

    batch = next(iter(train_loader))
    batch_imu, batch_gesture, batch_audio, batch_text, batch_scene, batch_joint_y, batch_intent_y, batch_scene_y = batch
    print(
        "[sanity] first batch "
        f"imu={tuple(batch_imu.shape)} "
        f"gesture={tuple(batch_gesture.shape)} "
        f"audio={tuple(batch_audio.shape)} "
        f"text={tuple(batch_text.shape)} "
        f"scene={tuple(batch_scene.shape)} "
        f"joint={tuple(batch_joint_y.shape)} "
        f"intent={tuple(batch_intent_y.shape)} "
        f"scene_y={tuple(batch_scene_y.shape)}"
    )

    model = Anchor2PerceiverIO(
        num_joint_classes=len(joint_class_names),
        num_intent_classes=len(intent_class_names),
        num_scene_classes=len(scene_class_names),
        joint_class_names=joint_class_names,
        model_dim=MODEL_DIM,
        num_latents=NUM_LATENTS,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        min_gate=MIN_GATE,
    ).to(DEVICE)
    with torch.no_grad():
        sanity_outputs = model(
            batch_imu.to(DEVICE),
            batch_gesture.to(DEVICE),
            batch_audio.to(DEVICE),
            batch_text.to(DEVICE),
            batch_scene.to(DEVICE),
        )
    print(f"[sanity] joint_logits shape {tuple(sanity_outputs['joint_logits'].shape)}")
    print(
        f"[sanity] avg gates "
        f"{dict(zip(MODALITY_KEYS, np.round(sanity_outputs['modality_gates'].mean(dim=0).cpu().numpy(), 4).tolist()))}"
    )

    joint_weights = base.build_class_weights(y_train_joint, len(joint_class_names))
    intent_weights = base.build_class_weights(y_train_raw, len(intent_class_names))
    scene_weights = base.build_class_weights(y_train_scene_raw, len(scene_class_names))
    if FOCAL_LOSS_GAMMA > 0.0:
        joint_criterion = WeightedFocalLoss(weight=joint_weights, gamma=FOCAL_LOSS_GAMMA)
        if FOCAL_LOSS_APPLY_AUX:
            intent_criterion = WeightedFocalLoss(weight=intent_weights, gamma=FOCAL_LOSS_GAMMA)
            scene_criterion = WeightedFocalLoss(weight=scene_weights, gamma=FOCAL_LOSS_GAMMA)
        else:
            intent_criterion = nn.CrossEntropyLoss(weight=intent_weights, label_smoothing=LABEL_SMOOTHING)
            scene_criterion = nn.CrossEntropyLoss(weight=scene_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        joint_criterion = nn.CrossEntropyLoss(weight=joint_weights, label_smoothing=LABEL_SMOOTHING)
        intent_criterion = nn.CrossEntropyLoss(weight=intent_weights, label_smoothing=LABEL_SMOOTHING)
        scene_criterion = nn.CrossEntropyLoss(weight=scene_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    checkpoint_path = OUTPUT_DIR / "improved_real_scene_anchor2.pt"
    scalers_path = OUTPUT_DIR / "scalers.pkl"
    label_encoder_path = OUTPUT_DIR / "label_encoder.pkl"
    report_path = OUTPUT_DIR / "classification_report.txt"
    intent_report_path = OUTPUT_DIR / "intent_classification_report.txt"
    scene_report_path = OUTPUT_DIR / "scene_classification_report.txt"
    metrics_path = OUTPUT_DIR / "metrics.json"
    loss_curve_path = OUTPUT_DIR / "loss_curve.png"
    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    intent_cm_path = OUTPUT_DIR / "intent_confusion_matrix.png"
    scene_cm_path = OUTPUT_DIR / "scene_confusion_matrix.png"
    scene_selection_path = OUTPUT_DIR / "scene_selection.json"
    gate_summary_path = OUTPUT_DIR / "modality_gates.json"
    modality_contribution_path = OUTPUT_DIR / "modality_contribution.json"
    modality_subset_metrics_path = OUTPUT_DIR / "modality_subset_metrics.json"

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_selection_score = -1.0
    best_epoch = 0
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    train_joint_losses: List[float] = []
    val_joint_losses: List[float] = []
    train_intent_losses: List[float] = []
    val_intent_losses: List[float] = []
    train_base_intent_losses: List[float] = []
    val_base_intent_losses: List[float] = []
    train_scene_losses: List[float] = []
    val_scene_losses: List[float] = []
    train_gesture_intent_losses: List[float] = []
    val_gesture_intent_losses: List[float] = []
    train_fallback_losses: List[float] = []
    val_fallback_losses: List[float] = []
    train_consistency_losses: List[float] = []
    train_margin_losses: List[float] = []
    train_missing_distill_losses: List[float] = []
    train_supcon_losses: List[float] = []

    print("[step] start training")
    optimizer_training_seconds = 0.0
    validation_seconds = 0.0
    epochs_completed = 0
    synchronize_device()
    training_loop_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        synchronize_device()
        train_epoch_start = time.perf_counter()
        train_loss, train_acc, train_parts = train_one_epoch(
            model,
            train_loader,
            joint_criterion,
            intent_criterion,
            scene_criterion,
            optimizer,
        )
        synchronize_device()
        optimizer_training_seconds += time.perf_counter() - train_epoch_start

        synchronize_device()
        validation_start = time.perf_counter()
        val_metrics = evaluate(
            model,
            val_loader,
            joint_criterion,
            intent_criterion,
            scene_criterion,
        )
        synchronize_device()
        validation_seconds += time.perf_counter() - validation_start
        epochs_completed = epoch

        train_losses.append(float(train_loss))
        val_losses.append(float(val_metrics["loss"]))
        train_accs.append(float(train_acc))
        val_accs.append(float(val_metrics["joint_acc"]))
        train_joint_losses.append(float(train_parts["joint_loss"]))
        val_joint_losses.append(float(val_metrics["joint_loss"]))
        train_intent_losses.append(float(train_parts["intent_loss"]))
        val_intent_losses.append(float(val_metrics["intent_loss"]))
        train_base_intent_losses.append(float(train_parts["base_intent_loss"]))
        val_base_intent_losses.append(float(val_metrics["base_intent_loss"]))
        train_scene_losses.append(float(train_parts["scene_loss"]))
        val_scene_losses.append(float(val_metrics["scene_loss"]))
        train_gesture_intent_losses.append(float(train_parts["gesture_intent_loss"]))
        val_gesture_intent_losses.append(float(val_metrics["gesture_intent_loss"]))
        train_fallback_losses.append(float(train_parts["fallback_loss"]))
        val_fallback_losses.append(float(val_metrics["fallback_loss"]))
        train_consistency_losses.append(float(train_parts["consistency_loss"]))
        train_margin_losses.append(float(train_parts["margin_loss"]))
        train_missing_distill_losses.append(float(train_parts["missing_distill_loss"]))
        train_supcon_losses.append(float(train_parts["supcon_loss"]))

        selection_score = (
            float(val_metrics["joint_acc"])
            + SELECTION_INTENT_WEIGHT * float(val_metrics["intent_acc"])
            + SELECTION_SCENE_WEIGHT * float(val_metrics["scene_acc"])
        )
        print(
            f"epoch {epoch:03d}/{EPOCHS:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_joint={val_metrics['joint_acc']:.4f} "
            f"val_intent={val_metrics['intent_acc']:.4f} val_scene={val_metrics['scene_acc']:.4f} "
            f"select={selection_score:.4f}"
        )

        improved = (selection_score > best_selection_score) or (
            np.isclose(selection_score, best_selection_score) and float(val_metrics["loss"]) < best_val_loss
        )
        if improved:
            best_selection_score = float(selection_score)
            best_val_acc = float(val_metrics["joint_acc"])
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_joint_classes": len(joint_class_names),
                    "model_dim": MODEL_DIM,
                    "num_latents": NUM_LATENTS,
                    "depth": DEPTH,
                    "num_heads": NUM_HEADS,
                    "dropout": DROPOUT,
                    "min_gate": MIN_GATE,
                    "imu_max_scale": IMU_MAX_SCALE,
                    "audio_max_scale": AUDIO_MAX_SCALE,
                    "intent_refine_scale": INTENT_REFINE_SCALE,
                    "gesture_logit_blend": GESTURE_LOGIT_BLEND,
                    "fallback_max_gate": FALLBACK_MAX_GATE,
                    "best_epoch": best_epoch,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "best_selection_score": best_selection_score,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[early_stop] no validation improvement for {PATIENCE} epochs")
                break

    synchronize_device()
    training_loop_seconds = time.perf_counter() - training_loop_start
    train_sample_count = int(len(y_train_joint))
    sample_exposures = int(train_sample_count * epochs_completed)
    avg_training_seconds_per_sample = safe_divide(
        optimizer_training_seconds,
        sample_exposures,
    )
    training_timing = {
        "optimizer_training_seconds": float(optimizer_training_seconds),
        "validation_seconds": float(validation_seconds),
        "training_loop_seconds": float(training_loop_seconds),
        "epochs_completed": int(epochs_completed),
        "train_sample_count": train_sample_count,
        "sample_exposures": sample_exposures,
        "avg_training_seconds_per_sample": avg_training_seconds_per_sample,
        "avg_training_seconds_per_sample_exposure": avg_training_seconds_per_sample,
        "cumulative_training_seconds_per_unique_sample": safe_divide(
            optimizer_training_seconds,
            train_sample_count,
        ),
        "avg_training_seconds_per_sample_definition": (
            "optimizer_training_seconds / (train_sample_count * epochs_completed)"
        ),
    }
    print(
        "[timing] "
        f"optimizer_train={optimizer_training_seconds:.3f}s "
        f"validation={validation_seconds:.3f}s "
        f"loop={training_loop_seconds:.3f}s "
        f"avg_train_per_sample={avg_training_seconds_per_sample:.8f}s"
    )

    if not checkpoint_path.exists():
        raise RuntimeError("Training finished without saving a checkpoint.")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        f"[best] epoch={checkpoint['best_epoch']} "
        f"val_acc={checkpoint['best_val_acc']:.4f} "
        f"val_loss={checkpoint['best_val_loss']:.4f} "
        f"selection={checkpoint.get('best_selection_score', checkpoint['best_val_acc']):.4f}"
    )

    val_metrics = evaluate(
        model,
        val_loader,
        joint_criterion,
        intent_criterion,
        scene_criterion,
    )
    base.save_loss_curve(train_losses, val_losses, loss_curve_path)

    if SKIP_TEST_EVAL:
        with open(scalers_path, "wb") as file:
            pickle.dump(scalers, file)
        with open(label_encoder_path, "wb") as file:
            pickle.dump(label_encoder, file)
        with open(scene_selection_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "train_videos": train_scene_selection,
                    "test_videos": {},
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        gate_summary = {
            "train_drop_probabilities": {
                "imu": IMU_DROP_PROB,
                "gesture": 0.0,
                "audio": AUDIO_DROP_PROB,
            },
            "support_scales": {
                "imu": IMU_MAX_SCALE,
                "gesture": 1.0,
                "audio": AUDIO_MAX_SCALE,
            },
            "validation_avg_gates": {
                MODALITY_DISPLAY_NAMES[key]: float(value)
                for key, value in zip(MODALITY_KEYS, val_metrics["avg_modality_gates"])
            },
        }
        with open(gate_summary_path, "w", encoding="utf-8") as file:
            json.dump(gate_summary, file, indent=2, ensure_ascii=False)

        metrics = {
            "timing": training_timing,
            "config": {
                "random_seed": base.RANDOM_SEED,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "model_dim": MODEL_DIM,
                "num_latents": NUM_LATENTS,
                "depth": DEPTH,
                "num_heads": NUM_HEADS,
                "dropout": DROPOUT,
                "device": str(DEVICE),
                "train_scene_source": "real",
                "skip_test_eval": True,
                "fusion_strategy": "gesture_text_scene_anchor_residual_perceiver_io",
                "label_smoothing": LABEL_SMOOTHING,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "min_gate": MIN_GATE,
                "intent_aux_weight": INTENT_AUX_WEIGHT,
                "base_intent_aux_weight": BASE_INTENT_AUX_WEIGHT,
                "scene_aux_weight": SCENE_AUX_WEIGHT,
                "gesture_intent_aux_weight": GESTURE_INTENT_AUX_WEIGHT,
                "intent_refine_scale": INTENT_REFINE_SCALE,
                "gesture_logit_blend": GESTURE_LOGIT_BLEND,
                "selection_intent_weight": SELECTION_INTENT_WEIGHT,
                "selection_scene_weight": SELECTION_SCENE_WEIGHT,
                "consistency_weight": CONSISTENCY_WEIGHT,
                "consistency_mask_prob": CONSISTENCY_MASK_PROB,
                "consistency_noise_std": CONSISTENCY_NOISE_STD,
                "consistency_temperature": CONSISTENCY_TEMPERATURE,
                "consistency_intent_weight": CONSISTENCY_INTENT_WEIGHT,
                "consistency_scene_weight": CONSISTENCY_SCENE_WEIGHT,
                "consistency_modalities": list(CONSISTENCY_MODALITIES),
                "margin_loss_weight": MARGIN_LOSS_WEIGHT,
                "margin_value": MARGIN_VALUE,
                "margin_intent_confusion_weight": MARGIN_INTENT_CONFUSION_WEIGHT,
                "margin_scene_confusion_weight": MARGIN_SCENE_CONFUSION_WEIGHT,
                "missing_distill_weight": MISSING_DISTILL_WEIGHT,
                "missing_distill_temperature": MISSING_DISTILL_TEMPERATURE,
                "missing_distill_intent_weight": MISSING_DISTILL_INTENT_WEIGHT,
                "missing_distill_scene_weight": MISSING_DISTILL_SCENE_WEIGHT,
                "missing_distill_force_mask": MISSING_DISTILL_FORCE_MASK,
                "missing_distill_modalities": list(MISSING_DISTILL_MODALITIES),
                "missing_distill_probabilities": MISSING_DISTILL_PROBS,
                "focal_loss_gamma": FOCAL_LOSS_GAMMA,
                "focal_loss_apply_aux": FOCAL_LOSS_APPLY_AUX,
                "fallback_max_gate": FALLBACK_MAX_GATE,
                "fallback_aux_weight": FALLBACK_AUX_WEIGHT,
                "supcon_loss_weight": SUPCON_LOSS_WEIGHT,
                "supcon_temperature": SUPCON_TEMPERATURE,
                "supcon_target": SUPCON_TARGET,
                "drop_probabilities": {
                    "imu": IMU_DROP_PROB,
                    "gesture": 0.0,
                    "audio": AUDIO_DROP_PROB,
                },
                "support_scales": {
                    "imu": IMU_MAX_SCALE,
                    "gesture": 1.0,
                    "audio": AUDIO_MAX_SCALE,
                },
            },
            "splits": {
                "train_samples": int(len(y_train_joint)),
                "val_samples": int(len(y_val_joint)),
                "test_samples": 0,
                "train_joint_distribution": base.summarize_joint_labels(y_train_joint_raw),
                "val_joint_distribution": base.summarize_joint_labels(y_val_joint_raw),
                "train_intent_distribution": base.summarize_labels(y_train_raw),
                "val_intent_distribution": base.summarize_labels(y_val_raw),
                "train_scene_distribution": base.summarize_scene_counts(
                    [base.SCENE_ID_TO_NAME[int(value)] for value in y_train_scene_raw.tolist()]
                ),
                "val_scene_distribution": base.summarize_scene_counts(
                    [base.SCENE_ID_TO_NAME[int(value)] for value in y_val_scene_raw.tolist()]
                ),
            },
            "best_checkpoint": {
                "epoch": int(checkpoint["best_epoch"]),
                "val_acc": float(checkpoint["best_val_acc"]),
                "val_loss": float(checkpoint["best_val_loss"]),
                "selection_score": float(checkpoint.get("best_selection_score", checkpoint["best_val_acc"])),
            },
            "final_metrics": {
                "val_loss": float(val_metrics["loss"]),
                "val_joint_acc": float(val_metrics["joint_acc"]),
                "val_intent_acc": float(val_metrics["intent_acc"]),
                "val_scene_acc": float(val_metrics["scene_acc"]),
                "val_joint_loss_only": float(val_metrics["joint_loss"]),
                "val_intent_loss_only": float(val_metrics["intent_loss"]),
                "val_base_intent_loss_only": float(val_metrics["base_intent_loss"]),
                "val_scene_loss_only": float(val_metrics["scene_loss"]),
                "val_gesture_intent_loss_only": float(val_metrics["gesture_intent_loss"]),
                "val_fallback_loss_only": float(val_metrics["fallback_loss"]),
            },
            "class_names": joint_class_names,
            "intent_class_names": intent_class_names,
            "scene_class_names": scene_class_names,
            "joint_class_weights": {
                joint_class_names[i]: float(joint_weights[i].item()) for i in range(len(joint_class_names))
            },
            "intent_class_weights": {
                intent_class_names[i]: float(intent_weights[i].item()) for i in range(len(intent_class_names))
            },
            "scene_class_weights": {
                scene_class_names[i]: float(scene_weights[i].item()) for i in range(len(scene_class_names))
            },
            "curves": {
                "train_loss": [float(value) for value in train_losses],
                "val_loss": [float(value) for value in val_losses],
                "train_acc": [float(value) for value in train_accs],
                "val_acc": [float(value) for value in val_accs],
                "train_joint_loss": [float(value) for value in train_joint_losses],
                "val_joint_loss": [float(value) for value in val_joint_losses],
                "train_intent_loss": [float(value) for value in train_intent_losses],
                "val_intent_loss": [float(value) for value in val_intent_losses],
                "train_base_intent_loss": [float(value) for value in train_base_intent_losses],
                "val_base_intent_loss": [float(value) for value in val_base_intent_losses],
                "train_scene_loss": [float(value) for value in train_scene_losses],
                "val_scene_loss": [float(value) for value in val_scene_losses],
                "train_gesture_intent_loss": [float(value) for value in train_gesture_intent_losses],
                "val_gesture_intent_loss": [float(value) for value in val_gesture_intent_losses],
                "train_fallback_loss": [float(value) for value in train_fallback_losses],
                "val_fallback_loss": [float(value) for value in val_fallback_losses],
                "train_consistency_loss": [float(value) for value in train_consistency_losses],
                "train_margin_loss": [float(value) for value in train_margin_losses],
                "train_missing_distill_loss": [float(value) for value in train_missing_distill_losses],
                "train_supcon_loss": [float(value) for value in train_supcon_losses],
            },
            "avg_modality_gates": {
                "validation": {
                    MODALITY_DISPLAY_NAMES[key]: float(value)
                    for key, value in zip(MODALITY_KEYS, val_metrics["avg_modality_gates"])
                },
            },
        }
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2, ensure_ascii=False)

        print("[saved]")
        print(f"  checkpoint      {checkpoint_path}")
        print(f"  scalers         {scalers_path}")
        print(f"  label_encoder   {label_encoder_path}")
        print(f"  metrics         {metrics_path}")
        print(f"  loss_curve      {loss_curve_path}")
        print(f"  scene_selection {scene_selection_path}")
        print(f"  gate_summary    {gate_summary_path}")
        return

    test_metrics = evaluate(
        model,
        test_loader,
        joint_criterion,
        intent_criterion,
        scene_criterion,
    )

    report = classification_report(
        test_metrics["joint_true"],
        test_metrics["joint_pred"],
        labels=np.arange(len(joint_class_names)),
        target_names=joint_class_names,
        zero_division=0,
        digits=4,
    )
    joint_true_names = label_encoder.inverse_transform(test_metrics["joint_true"])
    joint_pred_names = label_encoder.inverse_transform(test_metrics["joint_pred"])
    y_test_scene_true = np.array([base.split_joint_label(label_name)[0] for label_name in joint_true_names], dtype=object)
    y_test_scene_pred = np.array([base.split_joint_label(label_name)[0] for label_name in joint_pred_names], dtype=object)
    y_test_intent_true = np.array([base.split_joint_label(label_name)[1] for label_name in joint_true_names], dtype=object)
    y_test_intent_pred = np.array([base.split_joint_label(label_name)[1] for label_name in joint_pred_names], dtype=object)
    intent_report = classification_report(
        y_test_intent_true,
        y_test_intent_pred,
        labels=intent_class_names,
        target_names=intent_class_names,
        zero_division=0,
        digits=4,
    )
    scene_report = classification_report(
        y_test_scene_true,
        y_test_scene_pred,
        labels=scene_class_names,
        target_names=scene_class_names,
        zero_division=0,
        digits=4,
    )

    print("[test]")
    print(report)
    print("[intent_test]")
    print(intent_report)
    print("[scene_test]")
    print(scene_report)

    scene_name_to_idx = {name: index for index, name in enumerate(scene_class_names)}
    intent_name_to_idx = {name: index for index, name in enumerate(intent_class_names)}
    y_test_scene_true_idx = np.array([scene_name_to_idx[name] for name in y_test_scene_true], dtype=np.int64)
    y_test_scene_pred_idx = np.array([scene_name_to_idx[name] for name in y_test_scene_pred], dtype=np.int64)
    y_test_intent_true_idx = np.array([intent_name_to_idx[name] for name in y_test_intent_true], dtype=np.int64)
    y_test_intent_pred_idx = np.array([intent_name_to_idx[name] for name in y_test_intent_pred], dtype=np.int64)

    cm = base.save_confusion_matrix_plot(
        test_metrics["joint_true"],
        test_metrics["joint_pred"],
        joint_class_names,
        cm_path,
    )
    intent_cm = base.save_confusion_matrix_plot(
        y_test_intent_true_idx,
        y_test_intent_pred_idx,
        intent_class_names,
        intent_cm_path,
    )
    scene_cm = base.save_confusion_matrix_plot(
        y_test_scene_true_idx,
        y_test_scene_pred_idx,
        scene_class_names,
        scene_cm_path,
    )
    print("[step] modality contribution analysis on test split")
    modality_subset_metrics = evaluate_modality_subsets(
        model,
        test_features_scaled,
        y_test_joint,
        test_raw_labels,
        test_raw_scene_targets,
        joint_criterion,
        intent_criterion,
        scene_criterion,
    )
    modality_contribution = base.build_modality_contribution_report(modality_subset_metrics)
    modality_contribution_plot_paths = base.save_modality_contribution_bar_plots(
        modality_contribution,
        OUTPUT_DIR,
    )

    with open(scalers_path, "wb") as file:
        pickle.dump(scalers, file)
    with open(label_encoder_path, "wb") as file:
        pickle.dump(label_encoder, file)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)
    with open(intent_report_path, "w", encoding="utf-8") as file:
        file.write(intent_report)
    with open(scene_report_path, "w", encoding="utf-8") as file:
        file.write(scene_report)
    with open(scene_selection_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "train_videos": train_scene_selection,
                "test_videos": test_scene_selection,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    gate_summary = {
        "train_drop_probabilities": {
            "imu": IMU_DROP_PROB,
            "gesture": 0.0,
            "audio": AUDIO_DROP_PROB,
        },
        "support_scales": {
            "imu": IMU_MAX_SCALE,
            "gesture": 1.0,
            "audio": AUDIO_MAX_SCALE,
        },
        "validation_avg_gates": {
            MODALITY_DISPLAY_NAMES[key]: float(value)
            for key, value in zip(MODALITY_KEYS, val_metrics["avg_modality_gates"])
        },
        "test_avg_gates": {
            MODALITY_DISPLAY_NAMES[key]: float(value)
            for key, value in zip(MODALITY_KEYS, test_metrics["avg_modality_gates"])
        },
    }
    with open(gate_summary_path, "w", encoding="utf-8") as file:
        json.dump(gate_summary, file, indent=2, ensure_ascii=False)
    with open(modality_subset_metrics_path, "w", encoding="utf-8") as file:
        json.dump(modality_subset_metrics, file, indent=2, ensure_ascii=False)
    with open(modality_contribution_path, "w", encoding="utf-8") as file:
        json.dump(modality_contribution, file, indent=2, ensure_ascii=False)

    metrics = {
        "timing": training_timing,
        "config": {
            "random_seed": base.RANDOM_SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "model_dim": MODEL_DIM,
            "num_latents": NUM_LATENTS,
            "depth": DEPTH,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "device": str(DEVICE),
            "train_scene_source": "real",
            "test_scene_source": "real",
            "fusion_strategy": "gesture_text_scene_anchor_residual_perceiver_io",
            "label_smoothing": LABEL_SMOOTHING,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "min_gate": MIN_GATE,
            "intent_aux_weight": INTENT_AUX_WEIGHT,
            "base_intent_aux_weight": BASE_INTENT_AUX_WEIGHT,
            "scene_aux_weight": SCENE_AUX_WEIGHT,
            "gesture_intent_aux_weight": GESTURE_INTENT_AUX_WEIGHT,
            "intent_refine_scale": INTENT_REFINE_SCALE,
            "gesture_logit_blend": GESTURE_LOGIT_BLEND,
            "selection_intent_weight": SELECTION_INTENT_WEIGHT,
            "selection_scene_weight": SELECTION_SCENE_WEIGHT,
            "consistency_weight": CONSISTENCY_WEIGHT,
            "consistency_mask_prob": CONSISTENCY_MASK_PROB,
            "consistency_noise_std": CONSISTENCY_NOISE_STD,
            "consistency_temperature": CONSISTENCY_TEMPERATURE,
            "consistency_intent_weight": CONSISTENCY_INTENT_WEIGHT,
            "consistency_scene_weight": CONSISTENCY_SCENE_WEIGHT,
            "consistency_modalities": list(CONSISTENCY_MODALITIES),
            "margin_loss_weight": MARGIN_LOSS_WEIGHT,
            "margin_value": MARGIN_VALUE,
            "margin_intent_confusion_weight": MARGIN_INTENT_CONFUSION_WEIGHT,
            "margin_scene_confusion_weight": MARGIN_SCENE_CONFUSION_WEIGHT,
            "missing_distill_weight": MISSING_DISTILL_WEIGHT,
            "missing_distill_temperature": MISSING_DISTILL_TEMPERATURE,
            "missing_distill_intent_weight": MISSING_DISTILL_INTENT_WEIGHT,
            "missing_distill_scene_weight": MISSING_DISTILL_SCENE_WEIGHT,
            "missing_distill_force_mask": MISSING_DISTILL_FORCE_MASK,
            "missing_distill_modalities": list(MISSING_DISTILL_MODALITIES),
            "missing_distill_probabilities": MISSING_DISTILL_PROBS,
            "focal_loss_gamma": FOCAL_LOSS_GAMMA,
            "focal_loss_apply_aux": FOCAL_LOSS_APPLY_AUX,
            "fallback_max_gate": FALLBACK_MAX_GATE,
            "fallback_aux_weight": FALLBACK_AUX_WEIGHT,
            "supcon_loss_weight": SUPCON_LOSS_WEIGHT,
            "supcon_temperature": SUPCON_TEMPERATURE,
            "supcon_target": SUPCON_TARGET,
            "modality_contribution_method": "shapley_values_on_masked_test_subsets",
            "drop_probabilities": {
                "imu": IMU_DROP_PROB,
                "gesture": 0.0,
                "audio": AUDIO_DROP_PROB,
            },
            "support_scales": {
                "imu": IMU_MAX_SCALE,
                "gesture": 1.0,
                "audio": AUDIO_MAX_SCALE,
            },
        },
        "splits": {
            "train_samples": int(len(y_train_joint)),
            "val_samples": int(len(y_val_joint)),
            "test_samples": int(len(y_test_joint)),
            "train_joint_distribution": base.summarize_joint_labels(y_train_joint_raw),
            "val_joint_distribution": base.summarize_joint_labels(y_val_joint_raw),
            "test_joint_distribution": base.summarize_joint_labels(test_joint_labels_raw),
            "train_intent_distribution": base.summarize_labels(y_train_raw),
            "val_intent_distribution": base.summarize_labels(y_val_raw),
            "test_intent_distribution": base.summarize_labels(test_raw_labels),
            "train_scene_distribution": base.summarize_scene_counts(
                [base.SCENE_ID_TO_NAME[int(value)] for value in y_train_scene_raw.tolist()]
            ),
            "val_scene_distribution": base.summarize_scene_counts(
                [base.SCENE_ID_TO_NAME[int(value)] for value in y_val_scene_raw.tolist()]
            ),
            "test_scene_distribution": base.summarize_scene_counts(
                [base.SCENE_ID_TO_NAME[int(value)] for value in test_raw_scene_targets.tolist()]
            ),
        },
        "best_checkpoint": {
            "epoch": int(checkpoint["best_epoch"]),
            "val_acc": float(checkpoint["best_val_acc"]),
            "val_loss": float(checkpoint["best_val_loss"]),
            "selection_score": float(checkpoint.get("best_selection_score", checkpoint["best_val_acc"])),
        },
        "final_metrics": {
            "val_loss": float(val_metrics["loss"]),
            "val_joint_acc": float(val_metrics["joint_acc"]),
            "val_intent_acc": float(val_metrics["intent_acc"]),
            "val_scene_acc": float(val_metrics["scene_acc"]),
            "val_joint_loss_only": float(val_metrics["joint_loss"]),
            "val_intent_loss_only": float(val_metrics["intent_loss"]),
            "val_base_intent_loss_only": float(val_metrics["base_intent_loss"]),
            "val_scene_loss_only": float(val_metrics["scene_loss"]),
            "val_gesture_intent_loss_only": float(val_metrics["gesture_intent_loss"]),
            "val_fallback_loss_only": float(val_metrics["fallback_loss"]),
            "test_loss": float(test_metrics["loss"]),
            "test_joint_acc": float(test_metrics["joint_acc"]),
            "test_intent_acc": float(np.mean(y_test_intent_true == y_test_intent_pred)),
            "test_scene_acc": float(np.mean(y_test_scene_true == y_test_scene_pred)),
            "test_joint_loss_only": float(test_metrics["joint_loss"]),
            "test_intent_loss_only": float(test_metrics["intent_loss"]),
            "test_base_intent_loss_only": float(test_metrics["base_intent_loss"]),
            "test_scene_loss_only": float(test_metrics["scene_loss"]),
            "test_gesture_intent_loss_only": float(test_metrics["gesture_intent_loss"]),
            "test_fallback_loss_only": float(test_metrics["fallback_loss"]),
        },
        "class_names": joint_class_names,
        "intent_class_names": intent_class_names,
        "scene_class_names": scene_class_names,
        "joint_class_weights": {
            joint_class_names[i]: float(joint_weights[i].item()) for i in range(len(joint_class_names))
        },
        "intent_class_weights": {
            intent_class_names[i]: float(intent_weights[i].item()) for i in range(len(intent_class_names))
        },
        "scene_class_weights": {
            scene_class_names[i]: float(scene_weights[i].item()) for i in range(len(scene_class_names))
        },
        "curves": {
            "train_loss": [float(value) for value in train_losses],
            "val_loss": [float(value) for value in val_losses],
            "train_acc": [float(value) for value in train_accs],
            "val_acc": [float(value) for value in val_accs],
            "train_joint_loss": [float(value) for value in train_joint_losses],
            "val_joint_loss": [float(value) for value in val_joint_losses],
            "train_intent_loss": [float(value) for value in train_intent_losses],
            "val_intent_loss": [float(value) for value in val_intent_losses],
            "train_base_intent_loss": [float(value) for value in train_base_intent_losses],
            "val_base_intent_loss": [float(value) for value in val_base_intent_losses],
            "train_scene_loss": [float(value) for value in train_scene_losses],
            "val_scene_loss": [float(value) for value in val_scene_losses],
            "train_gesture_intent_loss": [float(value) for value in train_gesture_intent_losses],
            "val_gesture_intent_loss": [float(value) for value in val_gesture_intent_losses],
            "train_fallback_loss": [float(value) for value in train_fallback_losses],
            "val_fallback_loss": [float(value) for value in val_fallback_losses],
            "train_consistency_loss": [float(value) for value in train_consistency_losses],
            "train_margin_loss": [float(value) for value in train_margin_losses],
            "train_missing_distill_loss": [float(value) for value in train_missing_distill_losses],
            "train_supcon_loss": [float(value) for value in train_supcon_losses],
        },
        "avg_modality_gates": {
            "validation": {
                MODALITY_DISPLAY_NAMES[key]: float(value)
                for key, value in zip(MODALITY_KEYS, val_metrics["avg_modality_gates"])
            },
            "test": {
                MODALITY_DISPLAY_NAMES[key]: float(value)
                for key, value in zip(MODALITY_KEYS, test_metrics["avg_modality_gates"])
            },
        },
        "joint_confusion_matrix": cm.tolist(),
        "intent_confusion_matrix": intent_cm.tolist(),
        "scene_confusion_matrix": scene_cm.tolist(),
        "modality_contribution": modality_contribution,
    }
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("[saved]")
    print(f"  checkpoint      {checkpoint_path}")
    print(f"  scalers         {scalers_path}")
    print(f"  label_encoder   {label_encoder_path}")
    print(f"  report          {report_path}")
    print(f"  intent_report   {intent_report_path}")
    print(f"  scene_report    {scene_report_path}")
    print(f"  metrics         {metrics_path}")
    print(f"  loss_curve      {loss_curve_path}")
    print(f"  confusion_mat   {cm_path}")
    print(f"  intent_conf_mat {intent_cm_path}")
    print(f"  scene_conf_mat  {scene_cm_path}")
    print(f"  scene_selection {scene_selection_path}")
    print(f"  gate_summary    {gate_summary_path}")
    print(f"  modality_subset {modality_subset_metrics_path}")
    print(f"  modality_contrib {modality_contribution_path}")
    print(f"  modality_plot_joint  {modality_contribution_plot_paths['joint_acc']}")
    print(f"  modality_plot_intent {modality_contribution_plot_paths['intent_acc']}")
    print(f"  modality_plot_scene  {modality_contribution_plot_paths['scene_acc']}")


if __name__ == "__main__":
    main()
