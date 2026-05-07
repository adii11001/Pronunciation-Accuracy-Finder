import eng_to_ipa as ipa
import editdistance
from allosaurus.app import read_recognizer
import panphon
import panphon.distance

def normalize(word: str):
    return word.replace(" ", "").replace("̥", "")

target_word = "cricket bat"
model = read_recognizer()
expected_ipa = ipa.convert(target_word)[1:]
actual_ipa = model.recognize('enunciation.wav')

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

print(f"Expected: {clean_expected_ipa}")
print(f"Actual:   {clean_actual_ipa}")
print(f"Levenshtein Accuracy: {accuracy_score:.2f}%")
print(f"PanPhone Accuracy: {accuracy_score_panphon:.2f}%")
print(f"Weighted Score: {final_score:.2f}%")
