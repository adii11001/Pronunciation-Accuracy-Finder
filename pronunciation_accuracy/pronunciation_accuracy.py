import eng_to_ipa 
import epitran 
from pronunciation_accuracy.language_maps import EPITRAN_LANG_MAP, ALLOSAURUS_LANG_MAP
from allosaurus.app import read_recognizer
import editdistance
import panphon 
import panphon.distance
import warnings
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

def compute_score(actual_ipa, expected_ipa):
    max_len = max(len(actual_ipa), len(expected_ipa))
    min_len = min(len(actual_ipa), len(expected_ipa))

    # Distance calculations
    _panphon = panphon.distance.Distance()
    panphon_dist = _panphon.feature_edit_distance(actual_ipa, expected_ipa)
    levenshtein_dist = editdistance.eval(actual_ipa, expected_ipa)
    
    # Accuracy score calculations
    accuracy_score_panphon = ((max_len - panphon_dist) / max_len) * 100
    accuracy_score_levenshtein = ((max_len - levenshtein_dist) / max_len) * 100

    # Hybrid score calculation
    hybrid_score = 0.7 * accuracy_score_panphon + 0.3 * accuracy_score_levenshtein
    final_hybrid_score = hybrid_score * (min_len / max_len)
    return final_hybrid_score
