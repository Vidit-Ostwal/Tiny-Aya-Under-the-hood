# sensitivity_utils.py
# TIN-17: Layer Sensitivity Profiling with Direction Injection
#
# Core functions for computing language direction vectors and measuring
# layer-wise sensitivity via residual stream injection across Tiny Aya variants.
#
# Data loading is handled by src.data.flores_loader.load_flores_parallel_corpus.
# Language definitions are sourced from src.utils.languages.Language.

import torch
import numpy as np
from tqdm.auto import tqdm


def get_mean_hidden_states(sentences, model, tokenizer, device, max_length=64, batch_size=4):
    """
    Run sentences through the model and return mean-pooled hidden states
    at every layer (embedding layer + all transformer layers).

    Pooling is over non-padding tokens only, using the attention mask.
    Hidden states are moved to CPU immediately after each batch to avoid
    OOM — 16GB VRAM is tight with a 3.35B model loaded in float16.

    Args:
        sentences:   list of strings
        model:       loaded AutoModelForCausalLM (float16, device_map='auto')
        tokenizer:   corresponding AutoTokenizer
        device:      torch device string ('cuda' or 'cpu')
        max_length:  token limit per sentence (kept low to save VRAM)
        batch_size:  kept small (4) to avoid OOM with output_hidden_states=True

    Returns:
        np.ndarray of shape (n_layers + 1, hidden_dim)
        Index 0 = embedding layer output; indices 1..N = transformer layers.
        Values are the mean across all input sentences.
    """
    model.eval()
    all_hidden = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states  = outputs.hidden_states
        attention_mask = inputs["attention_mask"]

        for sentence_idx in range(len(batch)):
            mask = attention_mask[sentence_idx].unsqueeze(-1).float()
            n_tokens = mask.sum()
            per_layer = []
            for layer_hs in hidden_states:
                pooled = (layer_hs[sentence_idx] * mask).sum(0) / n_tokens
                per_layer.append(pooled.cpu().float().numpy())  # move to CPU immediately
            all_hidden.append(np.stack(per_layer))

        # Explicitly free GPU memory after every batch
        del outputs, hidden_states, inputs
        torch.cuda.empty_cache()

    stacked = np.stack(all_hidden)  # (n_sentences, n_layers+1, hidden_dim)
    return stacked.mean(axis=0)     # (n_layers+1, hidden_dim)


def compute_language_directions(corpus, english_key, model, tokenizer, device):
    """
    Compute a language direction vector at each layer for each language in corpus.

    Direction is defined as:
        d_i(L) = mean_pool(L, layer_i) - mean_pool(English, layer_i)

    Args:
        corpus:       dict[str, list[str]] as returned by
                      load_flores_parallel_corpus — keys are lang_name strings
        english_key:  the key in corpus corresponding to English (e.g. "english")
        model:        loaded AutoModelForCausalLM
        tokenizer:    corresponding AutoTokenizer
        device:       torch device string

    Returns:
        dict: lang_name -> np.ndarray of shape (n_layers+1, hidden_dim)
    """
    print("  Computing English mean hidden states...")
    english_mean = get_mean_hidden_states(corpus[english_key], model, tokenizer, device)

    directions = {}
    for lang_name, sentences in tqdm(corpus.items(), desc="  Computing directions"):
        if lang_name == english_key:
            continue
        lang_mean = get_mean_hidden_states(sentences, model, tokenizer, device)
        directions[lang_name] = lang_mean - english_mean  # (n_layers+1, hidden_dim)

    return directions


def make_injection_hook(direction_vector, scale, device):
    """
    Returns a PyTorch forward hook that injects a scaled language direction
    vector into the residual stream output of a transformer layer.

    The hook must be registered with register_forward_hook() and removed
    with handle.remove() after each forward pass to isolate layers.

    dtype=float16 to match the model computation dtype — using float32
    causes a dtype mismatch RuntimeError at runtime.

    Args:
        direction_vector: np.ndarray of shape (hidden_dim,)
        scale:            injection strength (alpha). Set to 20.0;
                          values above ~15 may cause non-linear saturation
                          (observed as sensitivity > 1.0 for Arabic at layer 5)
        device:           torch device string

    Returns:
        hook function compatible with nn.Module.register_forward_hook()
    """
    direction_tensor = torch.tensor(direction_vector, dtype=torch.float16).to(device)

    def hook(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            modified = hidden + scale * direction_tensor.unsqueeze(0).unsqueeze(0)
            return (modified,) + output[1:]
        else:
            return output + scale * direction_tensor.unsqueeze(0).unsqueeze(0)

    return hook


def cosine_similarity_np(a, b):
    """Cosine similarity between two 1D numpy arrays."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def get_pooled_final_states(inputs, model):
    """
    Run a batched forward pass and return mean-pooled final-layer hidden states.

    Args:
        inputs: tokenizer output dict (already on device)
        model:  loaded model

    Returns:
        np.ndarray of shape (batch_size, hidden_dim)
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    final_hs = outputs.hidden_states[-1]                    # (batch, seq, hidden)
    mask = inputs["attention_mask"].unsqueeze(-1).float()   # (batch, seq, 1)
    pooled = (final_hs * mask).sum(1) / mask.sum(1)         # (batch, hidden)
    return pooled.cpu().float().numpy()


def measure_sensitivity_batched(injection_sentences, direction_vectors, model,
                                 tokenizer, device, max_length=64, scale=20.0):
    """
    Batched injection sweep across all transformer layers.

    Registers a forward hook at each layer in turn, runs a single batched
    forward pass, and measures how much the final-layer representations
    changed vs baseline via (1 - cosine_similarity).

    Batching all injection sentences together reduces forward passes from
    n_sentences * n_layers to just n_layers (~20x speedup).

    Sensitivity interpretation:
        0.0  — injection had no effect on the final representation
        ~1.0 — final representation is nearly orthogonal to baseline
        >1.0 — representation pushed past orthogonality; likely saturation
               at the chosen scale (observed for Arabic at layer 5, scale=20)

    Args:
        injection_sentences: list of strings
        direction_vectors:   np.ndarray of shape (n_layers+1, hidden_dim)
                             index 0 = embedding layer
        model, tokenizer:    loaded model and tokenizer
        device:              torch device string
        max_length:          token limit per sentence
        scale:               injection scale (alpha)

    Returns:
        np.ndarray of shape (n_transformer_layers,)
    """
    model.eval()
    transformer_layers = model.model.layers
    n_layers = len(transformer_layers)

    # Tokenise all injection sentences into one batch
    inputs = tokenizer(
        injection_sentences, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length
    ).to(device)

    # Baseline: one forward pass with no injection
    baseline_reps = get_pooled_final_states(inputs, model)  # (n_sents, hidden_dim)

    sensitivity_scores = []

    for layer_idx in tqdm(range(n_layers), desc="    Sweeping layers", leave=False):
        # direction index is layer_idx+1 because index 0 is the embedding layer
        direction = direction_vectors[layer_idx + 1]
        hook_fn = make_injection_hook(direction, scale, device)
        handle = transformer_layers[layer_idx].register_forward_hook(hook_fn)

        injected_reps = get_pooled_final_states(inputs, model)
        handle.remove()  # must remove after each pass to isolate layers

        sims = [
            cosine_similarity_np(injected_reps[i], baseline_reps[i])
            for i in range(len(injection_sentences))
        ]
        sensitivity_scores.append(1.0 - np.mean(sims))

    return np.array(sensitivity_scores)
