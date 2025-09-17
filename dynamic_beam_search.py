import torch
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple

class EntropyDynamicBeamSearch:
    def __init__(
        self,
        model,
        tokenizer,
        max_length: int = 100,
        min_beams: int = 1,
        max_beams: int = 10,
        temperature: float = 1.0,
        early_stopping: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_beams = min_beams
        self.max_beams = max_beams
        self.temperature = temperature
        self.early_stopping = early_stopping
        self.vocab_size = len(tokenizer)
    
    def calculate_entropy_threshold(self, n: int) -> float:
        """
        Calculate entropy threshold for increasing beams to n.
        Formula: entropy(<0.8/n options> + <0.2/(vocab_size-n) options>)
        """
        if n <= 0 or n >= self.vocab_size:
            return float('inf')
        
        # Create probability distribution
        prob_dist = torch.zeros(self.vocab_size)
        
        # Top n tokens get 0.8/n probability each
        prob_dist[:n] = 0.8 / n
        
        # Remaining tokens get 0.2/(vocab_size-n) probability each
        prob_dist[n:] = 0.2 / (self.vocab_size - n)
        
        # Calculate entropy
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12))
        
        return entropy.item()
    
    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """Calculate entropy of the next token distribution."""
        probs = F.softmax(logits / self.temperature, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        return entropy.item()
    
    def determine_optimal_beams(self, current_entropy: float) -> int:
        """
        Determine the optimal number of beams based on current entropy.
        Returns the smallest n where entropy > threshold(n)
        """
        # Pre-calculate thresholds for all possible beam sizes
        thresholds = {}
        for n in range(self.min_beams, self.max_beams + 1):
            thresholds[n] = self.calculate_entropy_threshold(n)
        
        # Find the optimal number of beams
        optimal_beams = self.min_beams
        
        for n in range(self.min_beams, self.max_beams + 1):
            if current_entropy > thresholds[n]:
                optimal_beams = n
        
        return min(optimal_beams, self.max_beams)
    
    def beam_search(self, input_text: str, num_return_sequences: int = 1) -> List[str]:
        """
        Perform dynamic beam search generation.
        """
        # Encode input
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        batch_size = input_ids.size(0)
        
        # Initialize beams
        beams = [{
            'sequence': input_ids.clone(),
            'score': 0.0,
            'finished': False
        }]
        
        for step in range(self.max_length - len(input_ids[0])):
            # Get all active beams (not finished)
            active_beams = [beam for beam in beams if not beam['finished']]
            
            if not active_beams:
                break
            
            # Prepare input for model
            all_sequences = torch.cat([beam['sequence'] for beam in active_beams])
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(all_sequences)
                next_token_logits = outputs.logits[:, -1, :]
            
            # Calculate entropy for each beam
            entropies = []
            for i in range(len(active_beams)):
                beam_logits = next_token_logits[i]
                entropy = self.calculate_entropy(beam_logits)
                entropies.append(entropy)
            
            # Determine optimal beams for each active beam
            optimal_beams_per_beam = [
                self.determine_optimal_beams(entropy) for entropy in entropies
            ]
            
            # Generate new candidates
            new_beams = []
            
            for i, beam in enumerate(active_beams):
                beam_logits = next_token_logits[i]
                probs = F.softmax(beam_logits / self.temperature, dim=-1)
                
                # Get top k tokens based on optimal beam size
                k = optimal_beams_per_beam[i]
                topk_probs, topk_indices = torch.topk(probs, k)
                
                for j in range(k):
                    token_id = topk_indices[j].unsqueeze(0).unsqueeze(0)
                    new_sequence = torch.cat([beam['sequence'], token_id], dim=-1)
                    
                    # Calculate new score (log probability)
                    new_score = beam['score'] + torch.log(topk_probs[j] + 1e-12).item()
                    
                    # Check if sequence is finished (EOS token or max length)
                    is_finished = (token_id.item() == self.tokenizer.eos_token_id or 
                                  new_sequence.size(-1) >= self.max_length)
                    
                    new_beams.append({
                        'sequence': new_sequence,
                        'score': new_score,
                        'finished': is_finished
                    })
            
            # Sort beams by score and keep top ones
            new_beams.sort(key=lambda x: x['score'], reverse=True)
            beams = new_beams[:self.max_beams]
            
            # Early stopping if all beams are finished
            if self.early_stopping and all(beam['finished'] for beam in beams):
                break
        
        # Prepare final results
        beams.sort(key=lambda x: x['score'], reverse=True)
        results = []
        
        for i, beam in enumerate(beams[:num_return_sequences]):
            sequence = beam['sequence'][0].tolist()
            text = self.tokenizer.decode(sequence, skip_special_tokens=True)
            results.append({
                'text': text,
                'score': beam['score'],
                'length': len(sequence)
            })
        
        return results

# Example usage and testing
class MockModel:
    """Mock model for testing purposes"""
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def __call__(self, input_ids):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return type('obj', (object,), {'logits': logits})

def test_dynamic_beam_search():
    # Mock tokenizer
    class MockTokenizer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size
            self.eos_token_id = 2
        
        def encode(self, text, return_tensors='pt'):
            return torch.tensor([[1, 2, 3]])
        
        def decode(self, tokens, skip_special_tokens=True):
            return "decoded text"
    
    # Test the entropy threshold calculation
    tokenizer = MockTokenizer(vocab_size=1000)
    model = MockModel(vocab_size=1000)
    
    beam_search = EntropyDynamicBeamSearch(
        model=model,
        tokenizer=tokenizer,
        min_beams=1,
        max_beams=5
    )
    
    # Test entropy threshold calculation
    print("Entropy thresholds for different beam sizes:")
    for n in range(1, 6):
        threshold = beam_search.calculate_entropy_threshold(n)
        print(f"Beams: {n}, Threshold: {threshold:.4f}")
    
    # Test entropy calculation
    test_logits = torch.randn(1000)
    entropy = beam_search.calculate_entropy(test_logits)
    print(f"\nSample entropy: {entropy:.4f}")
    
    # Test beam determination
    optimal_beams = beam_search.determine_optimal_beams(entropy)
    print(f"Optimal beams for entropy {entropy:.4f}: {optimal_beams}")

if __name__ == "__main__":
    test_dynamic_beam_search()