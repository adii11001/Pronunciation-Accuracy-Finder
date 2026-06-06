import eng_to_ipa as ipa
import editdistance
from allosaurus.app import read_recognizer
import panphon.distance

def normalize(word: str):
    return word.replace(" ", "").replace("̥", "")

target_word = "cricket bat"
expected_ipa = ipa.convert(target_word)[1:]

model = read_recognizer("eng2102") # More language-specific than uni2020 (default)
actual_ipa = model.recognize('cricket_bat.wav')

# Normalized ipa string
clean_expected_ipa = normalize(expected_ipa)
clean_actual_ipa = normalize(actual_ipa)

# Levenshtein distance
distance = editdistance.eval(clean_expected_ipa, clean_actual_ipa)
max_len = max(len(clean_expected_ipa), len(clean_actual_ipa))

# PanPhone distance
panphon_distance = panphon.distance.Distance()
distance_panphon = panphon_distance.feature_edit_distance(clean_expected_ipa, clean_actual_ipa)
accuracy_score_panphon = ((max_len - distance_panphon) / max_len) * 100

accuracy_score = ((max_len - distance) / max_len) * 100

# Hybrid score
hybrid_score = accuracy_score * 0.4 + accuracy_score_panphon * 0.6
final_score = hybrid_score * (min(len(clean_expected_ipa), len(clean_actual_ipa))) / max(len(clean_expected_ipa), len(clean_actual_ipa))
