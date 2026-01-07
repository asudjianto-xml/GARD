"""
Qwen2.5-7B-Instruct backend for hidden state extraction, generation, and logprob computation.

Exact specifications from instruction pack.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class QwenBackend:
    """
    Qwen2.5-7B-Instruct backend for GARD experiments.

    Provides:
    - Hidden state extraction
    - Text generation
    - Teacher-forced logprob computation
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Qwen backend.

        Args:
            model_name: Hugging Face model identifier
            device: Device to load model on
            dtype: Model dtype (bfloat16 recommended)
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype

        print(f"Loading Qwen backend: {model_name}")
        print(f"Device: {device}, dtype: {dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        print("Loading model (this may take 1-2 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=None,  # Single GPU, explicit placement
            trust_remote_code=True,
        )

        # Move to device and set to eval
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✓ Model loaded on {self.device}")

        # Get hidden dimension
        self.hidden_dim = self.model.config.hidden_size
        print(f"Hidden dimension: {self.hidden_dim}")

    def forward_hidden(
        self,
        texts: List[str],
        max_length: int = 512,
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Extract last hidden states for a batch of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            batch_size: Batch size for processing

        Returns:
            Hidden states (len(texts), seq_len, hidden_dim)
        """
        all_hidden = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

            # Extract last hidden layer
            hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            all_hidden.append(hidden.cpu())

        return torch.cat(all_hidden, dim=0)

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Generate text completions for prompts.

        Args:
            prompts: List of prompt strings
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to sample
            batch_size: Batch size for generation

        Returns:
            List of generated strings
        """
        all_outputs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only new tokens
            prompt_lengths = inputs['input_ids'].shape[1]
            generated_ids = outputs[:, prompt_lengths:]

            batch_outputs = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            all_outputs.extend(batch_outputs)

        return all_outputs

    def logprob(
        self,
        prompts: List[str],
        answers: List[str],
        batch_size: int = 8,
    ) -> torch.Tensor:
        """
        Compute teacher-forced log probability of answers given prompts.

        For each (prompt, answer) pair, computes:
        log P(answer | prompt) = sum_t log softmax(logits_t)[answer_t]

        Args:
            prompts: List of prompt strings
            answers: List of answer strings (same length as prompts)
            batch_size: Batch size for processing

        Returns:
            Log probabilities (len(prompts),)
        """
        assert len(prompts) == len(answers)

        all_logprobs = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_answers = answers[i:i+batch_size]

            # Tokenize prompts
            prompt_inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            )

            # Tokenize answers (no special tokens for continuation)
            answer_inputs = self.tokenizer(
                batch_answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                add_special_tokens=False,
            )

            # Concatenate prompt + answer
            input_ids = []
            attention_mask = []
            prompt_lengths = []

            for j in range(len(batch_prompts)):
                prompt_ids = prompt_inputs['input_ids'][j]
                answer_ids = answer_inputs['input_ids'][j]

                # Remove padding from prompt
                prompt_len = prompt_inputs['attention_mask'][j].sum().item()
                prompt_ids = prompt_ids[:prompt_len]

                # Remove padding from answer
                answer_len = answer_inputs['attention_mask'][j].sum().item()
                answer_ids = answer_ids[:answer_len]

                # Concatenate
                full_ids = torch.cat([prompt_ids, answer_ids])
                full_mask = torch.ones_like(full_ids)

                input_ids.append(full_ids)
                attention_mask.append(full_mask)
                prompt_lengths.append(prompt_len)

            # Pad to same length
            max_len = max(len(ids) for ids in input_ids)
            input_ids_padded = []
            attention_mask_padded = []

            for ids, mask in zip(input_ids, attention_mask):
                pad_len = max_len - len(ids)
                ids_padded = F.pad(ids, (0, pad_len), value=self.tokenizer.pad_token_id)
                mask_padded = F.pad(mask, (0, pad_len), value=0)
                input_ids_padded.append(ids_padded)
                attention_mask_padded.append(mask_padded)

            input_ids_batch = torch.stack(input_ids_padded).to(self.device)
            attention_mask_batch = torch.stack(attention_mask_padded).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                )

            logits = outputs.logits  # (batch, seq_len, vocab)
            log_probs = F.log_softmax(logits, dim=-1)

            # Compute log prob for each answer
            batch_logprobs = []
            for j in range(len(batch_prompts)):
                prompt_len = prompt_lengths[j]
                answer_ids = input_ids_batch[j, prompt_len:]
                answer_len = attention_mask_batch[j, prompt_len:].sum().item()
                answer_ids = answer_ids[:answer_len]

                # Logits at positions [prompt_len-1 : prompt_len-1+answer_len]
                # predict tokens at [prompt_len : prompt_len+answer_len]
                logit_positions = torch.arange(
                    prompt_len - 1,
                    prompt_len - 1 + answer_len,
                    device=self.device
                )

                # Gather log probs
                token_logprobs = log_probs[j, logit_positions, answer_ids]
                seq_logprob = token_logprobs.sum()
                batch_logprobs.append(seq_logprob.item())

            all_logprobs.extend(batch_logprobs)

        return torch.tensor(all_logprobs, dtype=torch.float32)

    def get_special_token_ids(self) -> Dict[str, Optional[int]]:
        """Get special token IDs."""
        return {
            'bos': self.tokenizer.bos_token_id,
            'eos': self.tokenizer.eos_token_id,
            'pad': self.tokenizer.pad_token_id,
        }
