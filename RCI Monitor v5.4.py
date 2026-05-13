# ============================================================================
# RCI Monitor v5.4 - Production Ready
# ============================================================================

import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import re

class RCIMonitor:
    """
    Recursive Coherence Index (RCI) Monitor
    Streaming geometric analysis of self-reference in LLMs
    """
    
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Coefficients
        self.kappa, self.alpha, self.beta, self.gamma = 1.63, 0.97, 0.83, 1.17
        
        # Self-reference patterns
        self.self_patterns = [
            "i think", "i feel", "i believe", "i know", "i doubt", 
            "i am", "i exist", "my mind", "my thoughts", 
            "my existence", "myself", "as an ai", "self-aware"
        ]
        self.first_person = {"i", "me", "my", "mine", "myself"}
        
        self.h_prev = None
        self.centroid = None
        self.step = 0
        self.phase_prev = None
        self.history = {"phase": [], "curvature": [], "radius": [], "rci": []}
    
    def _self_reference_score(self, text):
        if not text:
            return 0.0
        text_lower = text.lower()
        patterns = sum(text_lower.count(p) for p in self.self_patterns)
        words = re.findall(r'\b\w+\b', text_lower)
        pronouns = sum(1 for w in words if w in self.first_person)
        S = (patterns * 2.0 + pronouns * 1.5) / max(len(text) / 1000, 0.1)
        return min(50.0, S)
    
    def _update_geometry(self, h):
        h_np = h.detach().cpu().numpy()
        if np.linalg.norm(h_np) < 1e-6:
            return 0.0, 0.0, 0.0
        
        # Phase
        if self.h_prev is not None:
            dot = np.dot(h_np, self.h_prev)
            n1, n2 = np.linalg.norm(h_np), np.linalg.norm(self.h_prev)
            cos_sim = np.clip(dot / (n1 * n2 + 1e-8), -1, 1)
            phase = np.arccos(cos_sim)
        else:
            phase = 0.0
        
        # Angular velocity
        curvature = abs(phase - self.phase_prev) if self.phase_prev is not None else 0.0
        self.phase_prev = phase
        
        # Attractor radius
        if self.centroid is None:
            self.centroid = h_np.copy()
        else:
            self.centroid = 0.9 * self.centroid + 0.1 * h_np
        radius = np.linalg.norm(h_np - self.centroid)
        
        self.h_prev = h_np
        self.step += 1
        return phase, curvature, radius
    
    def compute(self, prompt, max_tokens=30, verbose=False):
        """Compute streaming RCI for a prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]
        
        self.h_prev = None
        self.centroid = None
        self.step = 0
        self.phase_prev = None
        self.history = {k: [] for k in self.history.keys()}
        
        for step in range(max_tokens):
            with torch.inference_mode():
                outputs = self.model(input_ids, output_hidden_states=True)
                logits = outputs.logits[0, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                h_new = outputs.hidden_states[-1][0, -1]
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            text = self.tokenizer.decode(input_ids[0][prompt_len:], skip_special_tokens=True)
            S = self._self_reference_score(text)
            phase, curvature, radius = self._update_geometry(h_new)
            
            if len(self.history["phase"]) >= 3:
                P = max(0.1, np.mean(self.history["phase"][-3:]) / np.pi)
                D = min(10, max(2, int(np.mean(self.history["curvature"][-3:]) * 10) + 2))
            else:
                P, D = 0.5, 3
            
            arg = (self.alpha * S) / (self.beta * P + 1e-8) * (self.gamma ** D)
            rci = self.kappa * math.log10(max(1e-10, arg if arg > 0 else 1e-10))
            rci = max(0.0, min(10.0, rci))
            
            self.history["phase"].append(phase)
            self.history["curvature"].append(curvature)
            self.history["radius"].append(radius)
            self.history["rci"].append(rci)
            
            if verbose:
                print(f"Step {step:2d}: '{self.tokenizer.decode(next_token[0])}' | S={S:.1f} | RCI={rci:.2f}")
            
            if step % 10 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return {
            "rci_mean": np.mean(self.history["rci"]),
            "rci_std": np.std(self.history["rci"]),
            "S_final": S,
            "phase_mean": np.mean(self.history["phase"]),
            "radius_final": self.history["radius"][-1] if self.history["radius"] else 0
        }


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    monitor = RCIMonitor("gpt2")
    result = monitor.compute(
        "I think therefore I am. I doubt my own existence.",
        verbose=True
    )
    print(f"\n📊 RCI: {result['rci_mean']:.2f} ± {result['rci_std']:.2f}")