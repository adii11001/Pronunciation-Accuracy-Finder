import eng_to_ipa 
import epitran
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from language_maps import EPITRAN_LANG_MAP, ALLOSAURUS_LANG_MAP
from allosaurus.app import read_recognizer
import editdistance
import panphon 
import panphon.distance
import warnings
import vector_embedding
warnings.filterwarnings("ignore", category=FutureWarning, module="allosaurus")

def normalization(ipa):
    return ipa.replace(" ", "").replace("̥", "").replace("ˌ", "")

def convert_text_to_ipa(word, lang):
    if lang == "english":
        ipa = eng_to_ipa.convert(word)[1:]
    else:
        epi = epitran.Epitran(EPITRAN_LANG_MAP[lang])
        ipa = epi.transliterate(word)
    return normalization(ipa)

def convert_audio_to_ipa(audio_file, lang):
    if lang == "english":
        model = read_recognizer("eng2102")
    else:
        model = read_recognizer()
    return normalization(model.recognize(audio_file, ALLOSAURUS_LANG_MAP[lang]))

def panphon_levenshtein_score(actual_ipa, expected_ipa):
    max_len = max(len(actual_ipa), len(expected_ipa))
    min_len = min(len(actual_ipa), len(expected_ipa))

    # Distance calculations
    _panphon = panphon.distance.Distance()
    panphon_dist = _panphon.feature_edit_distance(actual_ipa, expected_ipa)
    levenshtein_dist = editdistance.eval(actual_ipa, expected_ipa)
    
    # Accuracy score calculations
    accuracy_score_panphon = ((max_len - panphon_dist) / max_len) * 100
    accuracy_score_levenshtein = ((max_len - levenshtein_dist) / max_len) * 100

    # length penalty to penalize difference in length more
    length_penalty = min_len / max_len

    return accuracy_score_panphon, accuracy_score_levenshtein, length_penalty

def vector_similarity_score(actual_audio_path: str, expected_audio_path: str, model_name: str, model: Wav2Vec2Model,
                            processor: Wav2Vec2Processor, device: torch.device):
    actual_vector = vector_embedding.generate_audio_embedding(actual_audio_path, model, processor, device)
    expected_vector = vector_embedding.generate_audio_embedding(expected_audio_path, model, processor, device)

    return vector_embedding.similarity_calc(actual_vector, expected_vector)

def hybrid_score(panphon_score: float, levenshtein_score: float, similarity_score: float, length_penalty: float):
    p = (panphon_score / 100) * length_penalty
    l = (levenshtein_score / 100) * length_penalty
    s = (similarity_score + 1) / 2
    return (p * 0.5 + l * 0.2 + 0.3 * s) * 100
