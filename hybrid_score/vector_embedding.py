import torch
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch.nn.functional as F

def model_processor_init(model_name: str, device: torch.device):
    return Wav2Vec2Model.from_pretrained(model_name).eval().to(device), Wav2Vec2Processor.from_pretrained(model_name)

def generate_audio_embedding(audio_path: str, model: Wav2Vec2Model, processor: Wav2Vec2Processor, device: torch.device):
    """
    Load an audio file and return a 768-dim embedding via wav2vec2-base.
    Args:
        audio_path: Path to .wav or .mp3 file.
    Returns:
        1D tensor of shape (768,).
    """
    target_sampling_rate = 16000

    # librosa.load returns (ndarray, sr); sr=16000 forces resampling
    speech_array, sampling_rate = librosa.load(path=audio_path, sr=target_sampling_rate) # Load audio file into floating point time series

    """
    processor performs resampling (if needed; demands 16000 Hz), normalization of audio file, creates attention masks.
    It also converts the input numpy ndarray into pytorch tensor
    """
    input_values = processor(audio=speech_array, sampling_rate=target_sampling_rate, return_tensors="pt").input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)

    """
    A hidden state is all the memory stored at any time step during the processing of the input data in a RNN. 
    """
    last_hidden_state = outputs.last_hidden_state

    """
    The output of the hidden state is of the following type: [batch_size, sequence_length, hidden_size] 
    sequence_length will depend on the size of the audio file. Two vectors can be compared only when their dimensions 
    are the same. hidden_size is 768. The shape will be something like [1, 400, 768]. To make the vectors similar in 
    dimensions, we take the mean and substitute it with of each of the 400 array. Finally: [1, 768]     
    """
    audio_embedding = torch.mean(last_hidden_state, dim=1).squeeze() # Finally squeeze to convert into an array of 768 values

    return audio_embedding

def similarity_calc(embedding_1: torch.Tensor, embedding_2: torch.Tensor):
    return F.cosine_similarity(embedding_1, embedding_2, dim=0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/wav2vec2-base-960h"
    model, processor = model_processor_init(model_name, device)

    actual_vector = generate_audio_embedding("./test_audio/apple.wav", model, processor, device)
    expected_vector = generate_audio_embedding("./test_audio/apple.wav", model, processor, device)

    print(similarity_calc(actual_vector, expected_vector))
