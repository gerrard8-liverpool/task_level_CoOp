from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/workspace/meta_prompt_1')
    args = parser.parse_args()
    trainer_path = Path(args.root) / 'third_party/CoOp_clean/trainers/coop_priorres.py'
    text = trainer_path.read_text(encoding='utf-8')

    if 'def compute_residual_stats(' not in text:
        marker = '    def forward(\n        self,\n        context_gates: torch.Tensor = None,\n        lambda_t: torch.Tensor = None,\n        context_gates_base: torch.Tensor = None,\n    ):'
        method = r'''
    def compute_residual_stats(
        self,
        context_gates: torch.Tensor = None,
        lambda_t: torch.Tensor = None,
        context_gates_base: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute prompt-residual norms for mechanism analysis.

        All returned tensors are detached scalars and are used only for logging.
        Safe formula:   ctx_eff = ctx + lambda_t * (a - a0) * u_ctx
        Legacy formula: ctx_eff = ctx + lambda_t * (a - 1)  * u_ctx
        """
        device = self.ctx.device
        ctx = self.ctx.detach().float()
        u_ctx = self.u_ctx.detach().float()
        eps = torch.tensor(1e-12, device=device, dtype=torch.float32)

        zero = torch.zeros((), device=device, dtype=torch.float32)
        ctx_norm = ctx.norm().clamp_min(eps)

        if (not self.use_context_gating) or context_gates is None or lambda_t is None:
            return {
                "residual_mode": torch.tensor(float(self.use_legacy_residual), device=device),
                "ctx_norm": ctx_norm.detach(),
                "residual_norm": zero,
                "relative_residual_norm": zero,
                "safe_residual_norm": zero,
                "legacy_residual_norm": zero,
                "safe_relative_residual_norm": zero,
                "legacy_relative_residual_norm": zero,
                "safe_raw_residual_norm": zero,
                "legacy_raw_residual_norm": zero,
                "legacy_init_bias_norm": zero,
                "legacy_init_bias_relative_norm": zero,
            }

        gate = context_gates.detach().to(device=device, dtype=torch.float32)
        if gate.dim() == 2:
            gate = gate.squeeze(0)

        if context_gates_base is None:
            gate_base = gate.detach()
        else:
            gate_base = context_gates_base.detach().to(device=device, dtype=torch.float32)
            if gate_base.dim() == 2:
                gate_base = gate_base.squeeze(0)

        lam = lambda_t.detach().to(device=device, dtype=torch.float32).view(-1)[0]

        def make_shift(centered_gate: torch.Tensor) -> torch.Tensor:
            if ctx.dim() == 2:
                return centered_gate.view(-1, 1) * u_ctx
            return centered_gate.view(1, -1, 1) * u_ctx

        safe_raw_shift = make_shift(gate - gate_base)
        legacy_raw_shift = make_shift(gate - 1.0)
        legacy_init_raw_shift = make_shift(gate_base - 1.0)

        safe_shift = lam * safe_raw_shift
        legacy_shift = lam * legacy_raw_shift
        actual_shift = legacy_shift if self.use_legacy_residual else safe_shift

        safe_norm = safe_shift.norm()
        legacy_norm = legacy_shift.norm()
        actual_norm = actual_shift.norm()
        legacy_init_bias_norm = legacy_init_raw_shift.norm()

        return {
            "residual_mode": torch.tensor(float(self.use_legacy_residual), device=device),
            "ctx_norm": ctx_norm.detach(),
            "residual_norm": actual_norm.detach(),
            "relative_residual_norm": (actual_norm / ctx_norm).detach(),
            "safe_residual_norm": safe_norm.detach(),
            "legacy_residual_norm": legacy_norm.detach(),
            "safe_relative_residual_norm": (safe_norm / ctx_norm).detach(),
            "legacy_relative_residual_norm": (legacy_norm / ctx_norm).detach(),
            "safe_raw_residual_norm": safe_raw_shift.norm().detach(),
            "legacy_raw_residual_norm": legacy_raw_shift.norm().detach(),
            "legacy_init_bias_norm": legacy_init_bias_norm.detach(),
            "legacy_init_bias_relative_norm": (legacy_init_bias_norm / ctx_norm).detach(),
        }

'''
        if marker not in text:
            raise SystemExit('Could not find insertion marker before PromptLearnerPriorRes.forward')
        text = text.replace(marker, method + marker)

    forward_marker = '''        prompts = self.prompt_learner(
            context_gates=adapter_out["a"],
            lambda_t=adapter_out["lambda_t"],
            context_gates_base=adapter_out.get("a0", None),
        )
'''
    forward_repl = forward_marker + '''        with torch.no_grad():
            adapter_out.update(
                self.prompt_learner.compute_residual_stats(
                    context_gates=adapter_out["a"],
                    lambda_t=adapter_out["lambda_t"],
                    context_gates_base=adapter_out.get("a0", None),
                )
            )
'''
    if 'compute_residual_stats(' in text and 'adapter_out.update(\n                self.prompt_learner.compute_residual_stats' not in text:
        if forward_marker not in text:
            raise SystemExit('Could not find CustomCLIPPriorRes.forward prompt marker')
        text = text.replace(forward_marker, forward_repl)

    summary_marker = '''            f"{prefix}top1_weight": self._scalarize(aux["top1_weight"]),
            f"{prefix}top4_weight_sum": self._scalarize(aux["top4_weight_sum"]),
        }
'''
    summary_repl = '''            f"{prefix}top1_weight": self._scalarize(aux["top1_weight"]),
            f"{prefix}top4_weight_sum": self._scalarize(aux["top4_weight_sum"]),
        }
        for key in [
            "residual_mode",
            "ctx_norm",
            "residual_norm",
            "relative_residual_norm",
            "safe_residual_norm",
            "legacy_residual_norm",
            "safe_relative_residual_norm",
            "legacy_relative_residual_norm",
            "safe_raw_residual_norm",
            "legacy_raw_residual_norm",
            "legacy_init_bias_norm",
            "legacy_init_bias_relative_norm",
        ]:
            if key in aux:
                summary[f"{prefix}{key}"] = self._scalarize(aux[key])
'''
    if 'legacy_init_bias_relative_norm' not in text.split('def _build_summary', 1)[1].split('def _set_trainable', 1)[0]:
        if summary_marker not in text:
            raise SystemExit('Could not find summary marker')
        text = text.replace(summary_marker, summary_repl)

    fb_marker = '''        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return summary
'''
    fb_repl = '''        self._append_analysis_csv(summary)

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return summary
'''
    if 'self._append_analysis_csv(summary)' not in text:
        if fb_marker not in text:
            raise SystemExit('Could not find forward_backward return marker')
        text = text.replace(fb_marker, fb_repl)

    method_marker = '    def _set_trainable(self, prompt_on: bool, meta_on: bool):\n'
    append_method = r'''
    def _append_analysis_csv(self, summary: Dict[str, float]):
        """Append per-batch mechanism statistics to analysis_stats.csv."""
        out_dir = getattr(self, "output_dir", None) or getattr(self.cfg, "OUTPUT_DIR", None) or "."
        os.makedirs(out_dir, exist_ok=True)
        csv_path = osp.join(out_dir, "analysis_stats.csv")

        fields = [
            "epoch",
            "batch_idx",
            "num_batches",
            "stage",
            "loss",
            "acc",
            "loss_cls",
            "lambda_t",
            "meff",
            "keff",
            "a0_mean",
            "a_mean",
            "delta_a_norm",
            "residual_mode",
            "ctx_norm",
            "residual_norm",
            "relative_residual_norm",
            "safe_residual_norm",
            "legacy_residual_norm",
            "safe_relative_residual_norm",
            "legacy_relative_residual_norm",
            "safe_raw_residual_norm",
            "legacy_raw_residual_norm",
            "legacy_init_bias_norm",
            "legacy_init_bias_relative_norm",
        ]

        row = dict(summary)
        row["epoch"] = int(self.epoch) + 1
        row["batch_idx"] = int(self.batch_idx) + 1
        row["num_batches"] = int(self.num_batches)

        need_header = not osp.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write(",".join(fields) + "\n")
            values = []
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, str):
                    values.append(value)
                else:
                    try:
                        values.append(f"{float(value):.10g}")
                    except Exception:
                        values.append("")
            f.write(",".join(values) + "\n")

'''
    if 'def _append_analysis_csv(' not in text:
        if method_marker not in text:
            raise SystemExit('Could not find marker before _set_trainable')
        text = text.replace(method_marker, append_method + method_marker)

    backup = trainer_path.with_suffix('.py.bak_residual_logging')
    if not backup.exists():
        backup.write_text(trainer_path.read_text(encoding='utf-8'), encoding='utf-8')
    trainer_path.write_text(text, encoding='utf-8')
    print(f'[OK] patched {trainer_path}')
    print(f'[OK] backup  {backup}')

if __name__ == '__main__':
    main()
