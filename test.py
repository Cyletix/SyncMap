from sklearn.metrics import normalized_mutual_info_score
import numpy as np

trueLabel = [0,0,0,0,0,1,1,1,1,1]
# Get labels predicted by SyncMap
learned_labels = [0,0,0,0,0,-1,-1,-1,-1,-1]

# Handle -1 labels (noise) by assigning them a new label
# noise_label = max(learned_labels) + 1
# learned_labels_cal = np.array([label if label != -1 else noise_label for label in learned_labels])


learned_labels_cal=[]
for i,label in enumerate(learned_labels):
    if label==-1:
        label=max(learned_labels) + 1+i
    learned_labels_cal.append(label)
learned_labels_cal=np.array(learned_labels_cal)

print(learned_labels_cal)

# Calculate NMI score
nmi_score = normalized_mutual_info_score(trueLabel, learned_labels_cal)
nmi_score = normalized_mutual_info_score(trueLabel, learned_labels)


print(trueLabel)
print(learned_labels)
print(learned_labels_cal)
print(nmi_score)





import numpy as np
from sklearn.metrics import mutual_info_score

def entropy(labels):
    """Calculate the entropy of a label distribution."""
    label_counts = np.bincount(labels)
    probabilities = label_counts / len(labels)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

def nmi_score(estimated_labels, true_labels):
    """Calculate the normalized mutual information (NMI) between estimated and true labels.
    
    NMI(\hat{Y}, Y) = 2 * I(\hat{Y}; Y) / (H(\hat{Y}) + H(Y))
    """
    # Calculate mutual information I(\hat{Y}; Y)
    mutual_info = mutual_info_score(estimated_labels, true_labels)

    # Calculate entropies H(\hat{Y}) and H(Y)
    entropy_estimated = entropy(estimated_labels)
    entropy_true = entropy(true_labels)

    # Calculate NMI using the provided formula
    if entropy_estimated + entropy_true == 0:
        return 0  # Avoid division by zero if both entropies are zero

    nmi = 2 * mutual_info / (entropy_estimated + entropy_true)
    return nmi

# Example usage
estimated_labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
true_labels = np.array([0,0,0,0,0,1,1,1,1,1])


nmi_result = nmi_score(estimated_labels, true_labels)
print(f"NMI Score: {nmi_result:.4f}")



nmi_result = nmi_score(estimated_labels, trueLabel)
print(f"NMI Score: {nmi_result:.4f}")
